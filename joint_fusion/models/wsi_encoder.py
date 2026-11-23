"""
Code to generate WSI embeddings using i) only inference from a pretrained model for earlyfusion, and ii) retraining parts of the model for joint fusion
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# WARNING: Must import here otherwise it will default to hf and will not have the right paths
env_path = Path.cwd() / ".env"
print("CWD:", Path.cwd())
print(".env path:", env_path)
print(".env exists:", env_path.exists())
load_dotenv(env_path)
print("ENV HF_HOME:", os.getenv("HF_HOME"))
print("ENV HUGGINGFACE_HUB_CACHE:", os.getenv("HUGGINGFACE_HUB_CACHE"))
print("ENV HF_TOKEN set:", bool(os.getenv("HF_TOKEN")))


# WARNING: Load models after .env variables are obtained
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, Normalize, Compose
import torchvision.transforms.functional as TF

import timm
from timm.models.vision_transformer import VisionTransformer
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import logging

logger = logging.getLogger(__name__)

# from pdb import set_trace

device = "cuda" if torch.cuda.is_available() else "cpu"

from huggingface_hub import login

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("Warning: No Hugging Face token provided. Authentication might fail.")

# logger.info("HF_HOME: %s", constants.HF_HOME)
# logger.info("HF_HUB_CACHE: %s", constants.HF_HUB_CACHE)
# logger.info("TMPDIR: %s", os.getenv("TMPDIR"))

from huggingface_hub import constants

print("HF_HOME:", constants.HF_HOME)
print("HF_HUB_CACHE:", constants.HF_HUB_CACHE)
print("TMPDIR:", os.getenv("TMPDIR"))


class LearnedWeightedPool(nn.Module):
    """
    A simple pooling mechanism for weighted averaging of tile embeddings.
    """

    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (n_tiles, embedding_dim)
        weights = torch.softmax(self.attention(x), dim=0)  # (n_tiles, 1)
        return (weights * x).sum(dim=0)  # (embedding_dim,)


class AttentionPool(nn.Module):
    """
    Attention pooling mechanism using a learnable query vector to attend to tile embeddings.
    """

    def __init__(self, dim):
        super().__init__()
        # query, key, and value projections
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim**-0.5  # scaling factor for dot product attention

    def forward(self, x):
        # x shape: (n_tiles, embedding_dim)

        # create a learnable query vector (1, embedding_dim)
        # this will attend to all tile embeddings
        query = self.q(torch.mean(x, dim=0, keepdim=True))  # (1, embedding_dim)

        # project tiles to keys and values
        keys = self.k(x)  # (n_tiles, embedding_dim)
        values = self.v(x)  # (n_tiles, embedding_dim)

        # compute attention scores
        # (1, embedding_dim) @ (embedding_dim, n_tiles) = (1, n_tiles)
        attention_scores = (query @ keys.transpose(-2, -1)) * self.scale

        # normalize attention scores
        attention_weights = torch.softmax(
            attention_scores, dim=-1
        )  # (1, n_tiles)  [ attention = softmax(Q.K/sqrt(dim))V]

        # Compute weighted sum of values
        # (1, n_tiles) @ (n_tiles, embedding_dim) = (1, embedding_dim)
        output = attention_weights @ values

        return output, attention_weights  # Return both output and attention weights


# create a custom dataset to prepare tile data for entering into the encoder
class CustomDatasetWSI(Dataset):
    def __init__(self, tiles):
        # "tiles" : list of tensors
        self.tiles = tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]

        if isinstance(tile, torch.Tensor) and tile.dim() == 4 and tile.shape[0] == 1:
            tile = tile.squeeze(0)

        return tile


def load_uni_model(model_name="UNI"):
    """
    Load the UNI model from the MahmoodLab huggingface repository.
    """
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": torch.nn.SiLU,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    model = timm.create_model(
        f"hf-hub:MahmoodLab/{model_name}",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    )
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    model.eval()
    return model, transform


# Use the pretrained Lunit-DINO model [Kang et al. (2023), "Benchmarking Self-Supervised Learning on Diverse Pathology Datasets"] for WSI feature extraction (trained on histopathology images)
# Lunit-DINO uses the ViT-S architecture with DINO for SSL
# Refer to Caron et al. (2021), "Emerging Properties in Self-Supervised Vision Transformers" for DINO implementation
# Note: change torch cache directory to a non-home location [export TORCH_HOME=./torch_cache/]
def get_pretrained_lunit(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


class WSIEncoder(nn.Module):
    """
    A wrapper class for WSI encoding using either Lunit-DINO or UNI.
    By default, the model is fully frozen.
    For joint_fusion, the last transformer block and norm layers are unfrozen
    (for partial retraining).
    """

    def __init__(
        self,
        wsi_fm,
        pooling,
        pretrained=True,
        progress=False,
        patch_size=16,
        tile_batch_size=64,
    ):
        super(WSIEncoder, self).__init__()
        self.wsi_fm = wsi_fm
        self.pooling = pooling
        self.tile_batch_size = tile_batch_size

        # need to resize transform for training otherwise 256 != 224
        # self.resize_transform = transforms.Resize((224, 224))

        if self.wsi_fm == "lunit_DINO":
            self.model = self.vit_small(
                pretrained, progress, "DINO_p16", patch_size=patch_size
            )
            transform_lunit = Compose(
                [
                    Resize((224, 224)),  # resize image to 224x224 for the model
                    # ToTensor(),
                    # normalization parameters for lunit DINO from
                    # https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights
                    Normalize(
                        mean=[0.70322989, 0.53606487, 0.66096631],
                        std=[0.21716536, 0.26081574, 0.20723464],
                    ),
                ]
            )

            self.transform = transform_lunit
            self.embed_dim = 384

            # freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False

            # unfreeze last transformer block & norm layer for joint fusion
            if __name__ != "__main__":
                num_blocks = len(self.model.blocks)
                print(f"Total number of transformer blocks in lunit DINO: {num_blocks}")
                last_block_idx = num_blocks - 1
                # unfreeze the last transformer block
                print(f"Unfreezing block{last_block_idx}")
                for param in self.model.blocks[last_block_idx].parameters():
                    param.requires_grad = True

                # unfreeze the final norm layer
                for param in self.model.norm.parameters():
                    param.requires_grad = True

        elif self.wsi_fm == "uni":
            self.model, _ = load_uni_model()
            self.embed_dim = self.model.num_features
            self.uni_normalize = Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

            # set_trace()

            # freeze all layers by default
            for param in self.model.parameters():
                param.requires_grad = False

            # for joint fusion, unfreeze the last transformer block and norm layer
            if __name__ != "__main__":
                # # check this: unfreeze the last transformer block (UNI has 24 blocks, indexed 0-23)
                # for param in self.model.blocks[23].parameters():
                #     param.requires_grad = True

                # get the number of transformer blocks
                num_blocks = len(self.model.blocks)
                print(f"Total number of transformer blocks in UNI: {num_blocks}")

                # unfreeze the last transformer block
                last_block_idx = num_blocks - 1
                print(f"Unfreezing block {last_block_idx}")
                for param in self.model.blocks[last_block_idx].parameters():
                    param.requires_grad = True

                # unfreeze the final norm layer
                for param in self.model.norm.parameters():
                    param.requires_grad = True

        else:
            raise ValueError(f"Unsupported WSI foundation model: {self.wsi_fm}")

        if self.pooling == "learned_weighted":
            self.attention_pool = LearnedWeightedPool(self.embed_dim)

        if self.pooling == "attention":
            self.attention_pool = AttentionPool(self.embed_dim)

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(
            "Number of trainable params for the histology model (from generate_wsi_embeddings.py): ",
            trainable_params,
        )

        # set the model to evaluation mode for early fusion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.pooling in {"attention", "learned_weighted"}:
            self.attention_pool.to(self.device)

    def vit_small(self, pretrained, progress, key, patch_size=16):
        model = VisionTransformer(
            img_size=224,
            patch_size=patch_size,
            embed_dim=384,
            num_heads=6,
            num_classes=0,
        )

        if pretrained:
            pretrained_url = get_pretrained_lunit(key)
            model.load_state_dict(
                torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
            )
        return model

    def get_wsi_embeddings(self, x_wsi):
        """
        Expected inputs
        - Tensor: [T, 3, H, W]  (best)
        - OR list of T tensors: each [3, H, W]  (ok)

        Returns:
        - slide_embedding: [embed_dim] (or [1, embed_dim] depending on your pooling)
        - if attention pooling: also return attention_weights
        """

        # 1) Normalize input type -> tiles tensor [T, 3, H, W]
        if isinstance(x_wsi, list):
            # list of [3,H,W]
            if len(x_wsi) == 0:
                raise ValueError("Empty tile list for patient")
            if not isinstance(x_wsi[0], torch.Tensor):
                raise TypeError(f"Expected list of tensors, got {type(x_wsi[0])}")
            tiles = torch.stack(x_wsi, dim=0)  # [T,3,H,W]
        elif isinstance(x_wsi, torch.Tensor):
            tiles = x_wsi
        else:
            raise TypeError(f"Unsupported x_wsi type: {type(x_wsi)}")

        # 2) Fix shapes defensively (but no more DP squeezing gymnastics)
        if tiles.dim() == 5 and tiles.size(0) == 1:
            # occasionally someone passes [1,T,3,H,W]
            tiles = tiles.squeeze(0)
        if tiles.dim() != 4:
            raise ValueError(f"Expected tiles [T,3,H,W], got {tuple(tiles.shape)}")

        # 3) Ensure channels are correct
        if tiles.size(1) != 3:
            raise ValueError(
                f"Expected RGB tiles with C=3, got C={tiles.size(1)} and shape {tuple(tiles.shape)}"
            )

        # 4) Move to device
        tiles = tiles.to(self.device)  # keep grads for saliency if needed

        # 5) Resize + normalize in batch
        tiles = TF.resize(tiles, [224, 224])  # works on [N,C,H,W]

        if self.wsi_fm == "uni":
            tiles = self.uni_normalize(tiles)
        elif self.wsi_fm == "lunit_DINO":
            tiles = self.transform(tiles)
        else:
            raise ValueError(f"Unsupported wsi_fm: {self.wsi_fm}")

        # 6) Encode in minibatches to avoid OOM
        # (tune bs for Polaris / your GPU mem)
        bs = self.tile_batch_size
        embs = []
        for i in range(0, tiles.size(0), bs):
            feats = self.model(tiles[i : i + bs])  # [b, embed_dim]
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)
            embs.append(feats)

        embeddings_tensor = torch.cat(embs, dim=0)  # [T, embed_dim]

        # 7) Pool across tiles
        if self.pooling in {"attention", "learned_weighted"}:
            slide_embedding, attention_weights = self.attention_pool(embeddings_tensor)
            return slide_embedding, attention_weights
        else:
            logger.warning("Using average pooling since no valid pooling selected")
            return embeddings_tensor.mean(dim=0)

    # ------------------------------------------------------------------
    # Split encode / pool, used ONLY by the chunked (memory-bounded) saliency
    # path. These duplicate the logic of get_wsi_embeddings (and must be kept
    # in sync with it) so the training/inference forward path above is left
    # completely untouched -- an identically-trained model is unchanged.
    #
    # The decomposition is exact because the encoder is applied INDEPENDENTLY
    # per tile, i.e. the i-th embedding is
    #     \( E_i = \mathrm{enc}(x_i) \),
    # a function of tile \(x_i\) ALONE. All cross-tile coupling lives later, in
    # the attention pool. This per-tile independence is what lets stage 2 of the
    # chunked saliency recompute \(E_i\) one chunk at a time without changing the
    # gradient (see compute_saliency_chunked in testing/test.py).
    # ------------------------------------------------------------------
    def _prepare_tiles(self, x_wsi):
        """Steps (1)-(5) of get_wsi_embeddings: type/shape normalize, move to
        device, resize to 224, and apply the FM-specific normalization."""
        if isinstance(x_wsi, list):
            if len(x_wsi) == 0:
                raise ValueError("Empty tile list for patient")
            if not isinstance(x_wsi[0], torch.Tensor):
                raise TypeError(f"Expected list of tensors, got {type(x_wsi[0])}")
            tiles = torch.stack(x_wsi, dim=0)
        elif isinstance(x_wsi, torch.Tensor):
            tiles = x_wsi
        else:
            raise TypeError(f"Unsupported x_wsi type: {type(x_wsi)}")

        if tiles.dim() == 5 and tiles.size(0) == 1:
            tiles = tiles.squeeze(0)
        if tiles.dim() != 4:
            raise ValueError(f"Expected tiles [T,3,H,W], got {tuple(tiles.shape)}")
        if tiles.size(1) != 3:
            raise ValueError(
                f"Expected RGB tiles with C=3, got shape {tuple(tiles.shape)}"
            )

        tiles = tiles.to(self.device)
        tiles = TF.resize(tiles, [224, 224])
        if self.wsi_fm == "uni":
            tiles = self.uni_normalize(tiles)
        elif self.wsi_fm == "lunit_DINO":
            tiles = self.transform(tiles)
        else:
            raise ValueError(f"Unsupported wsi_fm: {self.wsi_fm}")
        return tiles

    def encode_tiles(self, x_wsi):
        r"""Per-tile foundation-model features, *before* pooling.

        Returns \( E = [\,E_1, \dots, E_T\,] \) of shape ``[T, encoder_dim]`` with
        \( E_i = \mathrm{enc}(x_i) \). Mirrors steps (1)-(6) of
        get_wsi_embeddings; the per-tile encode loop already runs in batches of
        ``tile_batch_size``, so the only difference from training is that we stop
        before pooling and hand the embeddings back.
        """
        tiles = self._prepare_tiles(x_wsi)
        bs = self.tile_batch_size
        embs = []
        for i in range(0, tiles.size(0), bs):
            feats = self.model(tiles[i : i + bs])
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)
            embs.append(feats)
        return torch.cat(embs, dim=0)  # [T, encoder_dim]

    def pool_tile_features(self, tile_features):
        """Step (7) of get_wsi_embeddings, starting from per-tile features."""
        if self.pooling in {"attention", "learned_weighted"}:
            slide_embedding, attention_weights = self.attention_pool(tile_features)
            return slide_embedding, attention_weights
        return tile_features.mean(dim=0), None
