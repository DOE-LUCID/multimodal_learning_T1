"""
Code to generate WSI embeddings using i) only inference from a pretrained model for earlyfusion, and ii) retraining parts of the model for joint fusion
"""

import os
import torch
from pathlib import Path
from dotenv import load_dotenv
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, Normalize, Compose
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

# from pdb import set_trace

device = "cuda" if torch.cuda.is_available() else "cpu"

# ensure you're authenticated with huggingface to access the UNI model weights
env_path = Path.cwd() / ".env"
load_dotenv(env_path)

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("Warning: No Hugging Face token provided. Authentication might fail.")


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

        return output.squeeze(0), attention_weights.squeeze(
            0
        )  # Return both output and attention weights


# create a custom dataset to prepare tile data for entering into the encoder
class CustomDatasetWSI(Dataset):
    def __init__(self, tiles, wsi_fm, transform=None):
        # "tiles" : list of tensors
        self.tiles = tiles
        self.transform = transform
        self.wsi_fm = wsi_fm

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        if self.transform:
            # set_trace()
            # # convert tensor to PIL Image only for UNI model to make it compatible with the transform function from timm
            # if self.wsi_fm == 'uni' and isinstance(tile, torch.Tensor):
            #     set_trace()
            #     # tile.shape: torch.Size([200, 3, 256, 256])
            #     tile = transforms.ToPILImage()(tile)
            # tile = self.transform(tile)
            if not isinstance(tile, torch.Tensor):
                tile = self.transform(tile)

            if self.wsi_fm == "lunit_DINO":
                tile = self.transform(tile)

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
        wsi_fm="lunit_DINO",
        pooling="average",
        pretrained=True,
        progress=False,
        patch_size=16,
    ):
        super(WSIEncoder, self).__init__()
        self.wsi_fm = wsi_fm
        self.pooling = pooling

        # need to resize transform for training otherwise 256 != 224
        # self.resize_transform = transforms.Resize((224, 224))

        if self.wsi_fm == "lunit_DINO":
            self.model = self.vit_small(
                pretrained, progress, "DINO_p16", patch_size=patch_size
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
        # 'x_wsi' contain data from all tiles (list of tensors residing on the gpu)
        # len(x_wsi) = number of tiles per WSI
        # should get embeddings for each tile and pool them to get embeddings at the patient level
        dataset = CustomDatasetWSI(x_wsi, self.wsi_fm, transform=self.transform)
        tile_loader = DataLoader(
            dataset, batch_size=1, shuffle=False
        )  # check later why the batch size is hard-coded to 1
        embeddings = []

        # for joint fusion. Need to train parts of the model alongside models for other modalities and the downstream task
        for tiles in tile_loader:
            tiles = tiles.to(device)
            features = self.model(
                tiles.squeeze(0)
            )  # Get rid of the leading dim; the model expects [batch_size, n_channels, h, w]; probably tied to batch_size hardcoded to 1?
            embeddings.append(features.cpu())  # is moving to cpu really needed??

        embeddings_tensor = torch.stack(embeddings)
        # for joint fusion, the backprop will be through the combined embeddings to the inputs

        if self.pooling in {"attention", "learned_weighted"}:
            slide_embedding = self.attention_pool(
                embeddings_tensor.squeeze(1).to(device)
            )
        else:  # average pooling

            slide_embedding = torch.mean(embeddings_tensor, dim=0)
        return slide_embedding
