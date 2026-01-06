# code to generate WSI embeddings using i) only inference from a pretrained model for early fusion, and ii) retraining parts of the model for joint fusion
import os
import argparse
import csv
import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datasets import CustomDataset, HDF5Dataset
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
from timm.models.vision_transformer import VisionTransformer
from torchvision.io import read_image
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
from pdb import set_trace

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    attention pooling mechanism using a learnable query vector to attend to tile embeddings.
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
            if not isinstance(tile, torch.Tensor):
                tile = self.transform(tile)

            if self.wsi_fm == "lunit_DINO":
                tile = self.transform(tile)

        return tile


# ensure you're authenticated with huggingface to access the UNI model weights
env_path = Path.cwd() / ".env"
load_dotenv(env_path)

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("Warning: No Hugging Face token provided. Authentication might fail.")


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
    # model = timm.create_model(f"hf-hub:MahmoodLab/{model_name}", pretrained=True, **timm_kwargs)
    # model = timm.create_model(f"hf-hub:MahmoodLab/{model_name}", pretrained=True)
    # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
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
            # self.model = self.vit_small(pretrained, progress, key, patch_size=patch_size)
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

        elif self.wsi_fm == "uni":
            self.model, self.transform = load_uni_model()
            self.embed_dim = 1024  # get it from the last layer output automatically
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
        if __name__ == "__main__":
            self.model.eval()  # use train mode when imported as module (joint fusion), and in inference mode when directly ran (early fusion)
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

        if (
            __name__ == "__main__"
        ):  # for early fusion. Only carry out inference using the pretrained model
            # forward pass through the pretrained model to obtain the embeddings
            with torch.no_grad():  # using in evaluation mode for early fusion
                # loop over all tiles within a WSI and get the averaged embedding (each WSI should have 200 tiles)
                tile_index = 0
                for tiles in tile_loader:
                    tile_index += 1
                    # print(f"Loaded {tile_index} of {len(tile_loader)} tiles")
                    tiles = tiles.to(device)
                    # tiles.shape torch.Size([1, 200, 3, 224, 224])
                    features = self.model(
                        tiles.squeeze(0)
                    )  # get rid of the leading singleton dim (tied to the batch size?); the model expects [batch_size, n_channels, h, w]
                    embeddings.append(features.cpu().numpy())
            embeddings_array = np.array(embeddings)

            # combine all tile embeddings into a single numpy array
            if self.pooling in {"attention", "learned_weight"}:
                # convert to tensor for attention pooling
                embeddings_tensor = torch.tensor(embeddings_array).squeeze(1).to(device)
                slide_embedding = self.attention_pool(embeddings_tensor).cpu().numpy()
            elif self.pooling == "average":  # average pooling
                slide_embedding = np.mean(embeddings_array, axis=1)
            elif (
                self.pooling == "no_pooling"
            ):  # no pooling; get embeddings from all the tiles
                # embeddings_array.shape : (1, 1000, 1024)
                # so just getting rid of the redundant dimension here and not doing any pooling op
                slide_embedding = np.mean(embeddings_array, axis=0)

        else:  # for joint fusion. Need to train parts of the model alongside models for other modalities and the downstream task
            for batch_idx, tiles in enumerate(tile_loader):
                tiles = tiles.to(device)  # Or self.device if available
                # Debug: Log to spot the failing tile (remove after fix)
                # print(
                #     f"Tile batch {batch_idx}: shape {tiles.shape}, finite: {torch.isfinite(tiles).all().item()}"
                # )

                # Robust squeeze: Handle loader output variations safely
                if tiles.dim() == 5:  # e.g., [1, 1, C, H, W] → [1, C, H, W]
                    tiles = tiles.squeeze(0)
                elif tiles.dim() != 4:  # Expect [1, C, H, W]
                    raise ValueError(
                        f"Unexpected tiles shape in batch {batch_idx}: {tiles.shape}"
                    )

                features = self.model(
                    tiles
                )  # Model forward: expect [feat_dim] or similar
                # Debug: Check post-model
                # print(
                #     f"Features batch {batch_idx}: shape {features.shape}, finite: {torch.isfinite(features).all().item()}"
                # )
                embeddings.append(features)

            if len(embeddings) == 0:
                raise ValueError("No tiles processed—check dataset/loader")

            embeddings_tensor = torch.stack(embeddings)
            # for joint fusion, the backprop will be through the combined embeddings to the inputs
            if self.pooling in {"attention", "learned_weighted"}:
                slide_embedding = self.attention_pool(
                    embeddings_tensor.squeeze(1).to(self.device)
                )
            else:  # average pooling
                slide_embedding = torch.mean(embeddings_tensor, dim=0)
        return slide_embedding


# for inference for early fusion
# if this code is run directly, it only generates the embeddings (one embedding for each TCGA slide) and saves it as a dictionary
# Generates embeddings for all samples, and not only for the training dataset (the train/validation/test is done later at the prediction stage)
if __name__ == "__main__":
    # run in only inference mode for early fusion
    # df containing the samples with both WSI and rnaseq data (generated by trainer.py)
    mapping_df = pd.read_json("./mapping_df_31Jan_1000tiles.json", orient="index")
    # set_trace()
    # remove entries with anomalous time_to_death and 'days_to_last_followup' data
    # excluded_ids = ['TCGA-05-4395', 'TCGA-86-8281']  # contains anomalous time to event and censoring data
    excluded_ids = []
    mapping_df = mapping_df[~mapping_df.index.isin(excluded_ids)]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_wsi_path",
        type=str,
        # default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/TCGA_WSI/batch_corrected/processed_svs/tiles/256px_9.9x/combined_tiles/', # on laptop
        default="/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_20X_1000tiles/tiles/256px_128um/combined/",
        # default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_13july/tiles/256px_9.9x/combined/',
        # on Polaris
        # Use 'find TCGA-* -type f -print0 | xargs -0 -I {} cp {} combined/' within the tiles directory to create 'combined' to prevent errors due to large number of files (it takes a while)
        help="Path to input WSI tiles",
    )
    parser.add_argument(
        "--input_size_wsi", type=int, default=256, help="input_size for path images"
    )
    parser.add_argument(
        "--wsi_fm",
        type=str,
        default="uni",
        choices=["lunit_DINO", "uni"],
        help="WSI foundation model to use",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="no_pooling",
        choices=["average", "learned_weighted", "attention", "no_pooling"],
        help="Pooling method for tile embeddings",
    )
    opt = parser.parse_args()
    # set_trace()

    custom_dataset = CustomDataset(opt, mapping_df, mode="wsi")
    # custom_dataset = HDF5Dataset(opt,
    #                              h5_file='mapping_data.h5',
    #                              split='all',
    #                              mode='wsi')
    train_loader = torch.utils.data.DataLoader(
        dataset=custom_dataset,
        batch_size=1,
        shuffle=False,
    )

    encoder = WSIEncoder(wsi_fm=opt.wsi_fm, pooling=opt.pooling)

    # Initialize an empty dictionary to store TCGA IDs and embeddings
    # excluded_ids = ['TCGA-05-4395', 'TCGA-86-8281']  # contains anomalous time to event and censoring data
    excluded_ids = []
    patient_embeddings = {}

    # loop over all samples in batches
    # can do this in batches as in generating rnaseq embeddings
    for batch_idx, (tcga_id, time_to_event, event_occurred, x_wsi, x_omic) in enumerate(
        train_loader
    ):
        tcga_id = tcga_id[0]  # # Assuming tcga_id is a batch of size 1
        if tcga_id in excluded_ids:
            print(f"Skipped {tcga_id}")
            continue
        print(
            f"TCGA ID: {tcga_id}, batch_idx: {batch_idx}, out of {len(custom_dataset)}"
        )
        embeddings_slide = encoder.get_wsi_embeddings(
            x_wsi
        )  # slide level embedding for each patient
        # save the embeddings in a file for using in early fusion
        embeddings_list = (
            embeddings_slide.tolist()
            if isinstance(embeddings_slide, np.ndarray)
            else embeddings_slide
        )
        patient_embeddings[tcga_id] = embeddings_list
        # set_trace()
        # if batch_idx == 5:
        #     break

    # write out the embeddings to a json file
    # filename = "./WSI_embeddings_23july.json"

    filename = "./WSI_embeddings_uni_31Jan_1000tiles.json"
    with open(filename, "w") as file:
        json.dump(patient_embeddings, file)

    # reduce storage precision for smaller file size
    wsi_embs_rounded = patient_embeddings.applymap(
        lambda x: [round(val, 4) for val in x]
    )
    wsi_embs_rounded.to_json(
        "./WSI_embeddings_uni_31Jan.rounded.json", orient="columns"
    )

    set_trace()
    # save the slide level embeddings

    # rounded_embeddings = [round(float(x), 3) for x in patient_embeddings] # round to 3 decimal places to reduce the json file size
    # with open(filename, 'w') as file:
    #     json.dump(rounded_embeddings, file)
