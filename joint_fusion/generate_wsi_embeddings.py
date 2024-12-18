# code to train models, or use pretrained models to generate WSI embeddings
import os
import argparse
import csv
import numpy as np
import pandas as pd
import torch
import json
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datasets import CustomDataset, HDF5Dataset
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from torchvision.io import read_image
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
from pdb import set_trace

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# create a custom dataset to prepare tile data for entering into the encoder
class CustomDatasetWSI(Dataset):
    def __init__(self, tiles, transform=None):
        # "tiles" : list of tensors
        self.tiles = tiles
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        if self.transform:
            tile = self.transform(tile)
        return tile


transform = Compose([
    Resize((224, 224)),  # resize image to 224x224 for the model
    # ToTensor(),
    # normalization parameters for lunit DINO from
    # https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights
    Normalize(mean=[0.70322989, 0.53606487, 0.66096631],
              std=[0.21716536, 0.26081574, 0.20723464]),
])


# dataset = CustomDataset(
#     image_dir='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/TCGA_WSI/batch_corrected/processed_svs/tiles/256px_9.9x/combined_tiles/',
#     transform=transform
# )
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # set shuffle to False for inference


# Use the pretrained Lunit-DINO model [Kang et al. (2023), "Benchmarking Self-Supervised Learning on Diverse Pathology Datasets"] for WSI feature extraction (trained on histopathology images)
# Lunit-DINO uses the ViT-S architecture with DINO for SSL
# Refer to Caron et al. (2021), "Emerging Properties in Self-Supervised Vision Transformers" for DINO implementation


# Note: change torch cache directory to a non-home location [export TORCH_HOME=./torch_cache/]
def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


class WSIEncoder(nn.Module):
    def __init__(self, pretrained=True, progress=False, key="DINO_p16", patch_size=16):
        super(WSIEncoder, self).__init__()
        self.model = self.vit_small(pretrained, progress, key, patch_size=patch_size)

        # print(self.model)  # print the entire model architecture
        # # print all children layers
        # for name, layer in self.model.named_children():
        #     print(name, layer)
        # # print all modules
        # for name, module in self.model.named_modules():
        #     print(name, module)
        #
        # for name, param in self.model.named_parameters():
        #     # if "head" not in name:
        #     param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of trainable params: ", trainable_params)
        # need to fix this when early fusion is integrated. Right, early fusion is carried out generating the embeddings separately, and using a different code for the fusion ('early_fusion_poc_combine_test_validation.py')
        if __name__ == "__main__":
            self.model.eval()  # use train mode when imported as module, and in inference model when directly ran
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        # total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print(f"Total number of trainable parameters in the model: {total_params / 1e6}M")

    def vit_small(self, pretrained, progress, key, patch_size=16):
        model = VisionTransformer(
            img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
        )

        # freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        if __name__ != "__main__":  # i.e. for joint fusion
            # unfreeze the last transformer block (block 11)
            for param in model.blocks[11].parameters():
                param.requires_grad = True

            # unfreeze the final norm layer
            for param in model.norm.parameters():
                param.requires_grad = True

        # print the trainable params
        for name, param in model.named_parameters():
            print(f"{name}: {param.requires_grad}")

        if pretrained:
            pretrained_url = get_pretrained_url(key)
            model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url, progress=progress))
        return model

    def get_wsi_embeddings(self, x_wsi):
        # 'x_wsi' contain data from all tiles (list of tensors residing on the gpu)
        # len(x_wsi) = number of tiles per WSI
        # should get embeddings for each tile and average them to get embeddings at the patient level
        dataset = CustomDatasetWSI(x_wsi, transform=transform)
        tile_loader = DataLoader(dataset, batch_size=1,
                                 shuffle=False)  # check later why the batch size is hard-coded to 1
        embeddings = []
        if __name__ == "__main__":
            # forward pass through the pretrained model to obtain the embeddings
            with torch.no_grad():  # using in evaluation mode for early fusion
                # loop over all tiles within a WSI and get the averaged embedding (each WSI should have 200 tiles)
                tile_index = 0
                for tiles in tile_loader:
                    tile_index += 1
                    # print(f"Loaded {tile_index} of {len(tile_loader)} tiles")
                    tiles = tiles.to(device)
                    # set_trace()
                    features = self.model(tiles.squeeze(0))  # get rid of the leading dim; the model expects [batch_size, n_channels, h, w]
                    # features = self.model(tiles)
                    embeddings.append(features.cpu().numpy())
            # set_trace()
            embeddings_array = np.array(embeddings)
            # Concatenate all tile embeddings into a single numpy array
            averaged_embeddings = np.mean(embeddings_array, axis=0)
        else:
            for tiles in tile_loader:
                tiles = tiles.to(device)
                features = self.model(tiles.squeeze(0))  # Get rid of the leading dim; the model expects [batch_size, n_channels, h, w]; probably tied to batch_size hardcoded to 1?
                embeddings.append(features.cpu())  # is moving to cpu really needed??
            embeddings_tensor = torch.stack(embeddings)
            # average the embeddings across the tile dimension
            averaged_embeddings = torch.mean(embeddings_tensor, dim=0)

            # # instead of storing all embeddings in a list, accumulate the sum of embeddings directly to prevent OOM error
            # total_embeddings = None
            # count = 0
            #
            # for tiles in tile_loader:
            #     tiles = tiles.to(device)
            #     features = self.model(tiles.squeeze(0))  # get rid of the leading dim; the model expects [batch_size, n_channels, h, w]
            #
            #     if total_embeddings is None:
            #         total_embeddings = features
            #     else:
            #         total_embeddings += features
            #     count += 1
            #
            # averaged_embeddings = total_embeddings / count

        return averaged_embeddings


# if this code is run directly, just generate the embeddings (one embedding for each TCGA sample) and save as a dictionary
# Generates embeddings for all samples, and not only for the training dataset (the train/validation/test is done later at the prediction stage)
if __name__ == "__main__":
    # run in only inference model for early fusion
    # df containing the samples with both WSI and rnaseq data (generated vy trainer.py)
    mapping_df = pd.read_json(
        "./mapping_df.json",
        orient='index')
    # remove entries with anomalous time_to_death and 'days_to_last_followup' data
    excluded_ids = ['TCGA-05-4395', 'TCGA-86-8281']  # contains anomalous time to event and censoring data
    mapping_df = mapping_df[~mapping_df.index.isin(excluded_ids)]

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wsi_path', type=str,
                        # default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/TCGA_WSI/batch_corrected/processed_svs/tiles/256px_9.9x/combined_tiles/', # on laptop
                        # default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_otsu_B/tiles/256px_9.9x/combined_tiles/',
                        default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_13july/tiles/256px_9.9x/combined/',
                        # on Polaris
                        # Use 'find TCGA-* -type f -print0 | xargs -0 -I {} cp {} combined/' within the tiles directory to create 'combined' to prevent errors due to large number of files (it takes a while)
                        help='Path to input WSI tiles')
    parser.add_argument('--input_size_wsi', type=int, default=256, help="input_size for path images")
    opt = parser.parse_args()

    custom_dataset = CustomDataset(opt,
                                   mapping_df,
                                   mode='wsi')
    # custom_dataset = HDF5Dataset(opt,
    #                              h5_file='mapping_data.h5',
    #                              split='all',
    #                              mode='wsi')
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                               batch_size=1,
                                               shuffle=False, )

    encoder = WSIEncoder()

    # Initialize an empty dictionary to store TCGA IDs and embeddings
    excluded_ids = ['TCGA-05-4395', 'TCGA-86-8281']  # contains anomalous time to event and censoring data
    patient_embeddings = {}
    # for batch_idx, (tcga_id, days_to_death, days_to_last_followup, event_occurred, x_wsi, x_omic) in enumerate(train_loader):
    # for batch_idx, (tcga_id, time_to_event, event_occurred, x_wsi, x_omic) in enumerate(train_loader):
    #     tcga_id = tcga_id[0] # # Assuming tcga_id is a batch of size 1
    #     print(f"TCGA ID: {tcga_id}, batch_idx: {batch_idx}, out of {len(custom_dataset)}")
    #     embeddings = encoder.get_wsi_embeddings(x_wsi) # tile averaged embedding for each patient
    #     # save the embeddings in a file for using in early fusion
    #     embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
    #     patient_embeddings[tcga_id] = embeddings_list
    #     # if batch_idx == 5:
    #     #     break

    # loop over all samples in batches
    for batch_idx, (tcga_id, time_to_event, event_occurred, x_wsi, x_omic) in enumerate(train_loader):
        tcga_id = tcga_id[0]  # # Assuming tcga_id is a batch of size 1
        if tcga_id in excluded_ids:
            print(f"Skipped {tcga_id}")
            continue
        print(f"TCGA ID: {tcga_id}, batch_idx: {batch_idx}, out of {len(custom_dataset)}")
        embeddings = encoder.get_wsi_embeddings(x_wsi)  # tile averaged embedding for each patient
        # save the embeddings in a file for using in early fusion
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        patient_embeddings[tcga_id] = embeddings_list
        # if batch_idx == 5:
        #     break

        # # loop over each batch (1 for each TCGA sample)
        # # x_wsi is a list of 200 tiles per WSI
        # for i in range(len(tcga_ids)):
        #     tcga_id = tcga_ids[i]
        #     print(f"TCGA ID: {tcga_id}, batch_idx: {batch_idx}, out of {len(train_loader)}")
        #     # set_trace()
        #     embeddings = encoder.get_wsi_embeddings(x_wsi[i])  # process each WSI individually (should loop over all tiles and get the tile-averaged embeddings)
        #     embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        #     patient_embeddings[tcga_id] = embeddings_list

    # filename = "./WSI_embeddings_23july.json"
    filename = "./WSI_embeddings_1sep.json"
    with open(filename, 'w') as file:
        json.dump(patient_embeddings, file)

        # set_trace()
        # 'x_wsi' contain data from all tiles (list of tensors residing on the gpu)
        # len(x_wsi) = number of tiles per WSI
        # should get embeddings for each tile and average them to get embeddings at the patient level
        # dataset = CustomDatasetWSI(x_wsi, transform=transform)
        # tile_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        # embeddings = []
        # # forward pass through the pretrained model to obtain the embeddings
        # with torch.no_grad():
        #     # loop over all tiles within a WSI and get the averaged embedding
        #     for tiles in tile_loader:
        #         # set_trace()
        #         tiles = tiles.to(device)
        #         features = WSIEncoder.model(
        #             tiles.squeeze(0))  # get rid of the leading dim; the model expects [batch_size, n_channels, h, w]
        #         embeddings.append(features.cpu().numpy())
        # embeddings_array = np.array(embeddings)
        # # Concatenate all tile embeddings into a single numpy array
        # averaged_embeddings = np.mean(embeddings_array, axis=0)
        # # # embeddings = np.concatenate(embeddings, axis=0)
        # # embeddings = np.mean(np.vstack(embeddings), axis=1)
        # # return averaged_embeddings
        # # main()
