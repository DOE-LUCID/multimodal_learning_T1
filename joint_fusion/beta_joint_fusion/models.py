import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import time
from generate_rnaseq_embeddings import get_omic_embeddings
from lookup_embeddings import (
    early_fusion_get_omic_embeddings,
    early_fusion_get_wsi_embeddings,
)
from generate_wsi_embeddings import WSIEncoder
import torchvision.models as models
from pdb import set_trace
from torchvision.models.vision_transformer import vit_b_32
from torchvision.models import ViT_B_32_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mlp_layers(embedding_dim, num_layers, base_dim=256):

    if num_layers < 2:
        raise ValueError("MLP Layers cannot be less than 2")
    elif num_layers == 2:
        return [int(embedding_dim), 1]

    # append the first layer
    mlp_layers = [int(embedding_dim)]

    for i in range(num_layers - 2):

        mlp_layers.append(int(base_dim * (1 / 2) ** i))

    # append the last layer
    mlp_layers.append(1)

    if len(mlp_layers) != num_layers:
        raise ValueError(
            "get_mlp_layers method failed to obtain mlp layers with the given num_layers"
        )

    return mlp_layers


# make the output(embedding) dimension a hyperparameter
class WSINetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(WSINetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_cnn = False
        self.use_resnet = False
        self.use_lunit_dino = True

        if self.use_cnn:

            self.net = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, embedding_dim),
                nn.ReLU(),
            )

        elif self.use_resnet:
            # 18 layer resnet
            # can train the whole model or freeze certain layers
            resnet18 = models.resnet18(pretrained=True)
            # remove the fully connected layer (classifier) and the final pooling layer
            # extract the final set of features
            # https://stackoverflow.com/questions/55083642/extract-features-from-last-hidden-layer-pytorch-resnet18
            layers = list(resnet18.children())[:-2]
            num_features_extracted = 512  # fixed for resnet18

            # self.net = nn.Sequential(
            #     *layers,
            #     nn.AdaptiveAvgPool2d((1, 1)),
            #     nn.Flatten(),
            #     nn.Linear(num_features_extracted, embedding_dim),
            #     nn.ReLU()
            # )
            # # set_trace()

            # vision transformer vit_b_32 arch ('base' version with 32 x 32 patches)
            vit = vit_b_32(weights=ViT_B_32_Weights)
            layers = list(vit.children())  # get all the layers
            vit_top = nn.Sequential(
                *layers[:-2]
            )  # remove the normalization and the pooling layer
            self.feature_extractor = nn.Sequential(*layers)
            num_features = 768
            # self.embedding_layer = nn.Linear(num_features, embedding_dim)
            self.net = nn.Sequential(
                *layers, nn.Flatten(), nn.Linear(num_features, embedding_dim)
            )

        elif self.use_lunit_dino:
            self.encoder = WSIEncoder(pretrained=True)
            self.net = nn.Sequential(
                nn.Linear(
                    384, embedding_dim
                ),  # to match the embedding dimension to the vit output
                nn.ReLU(),
            )

    def forward(self, x_wsi):
        # print("+++++++++++++ Input shape within WSINetwork: ", x_wsi.shape)
        if self.use_lunit_dino:
            embeddings = self.encoder.get_wsi_embeddings(x_wsi)
            embeddings = embeddings.to(self.encoder.device)
            return self.net(embeddings)

            # fixed
            # batch_embeddings = []

            # for patient_idx, patient_tiles in enumerate(x_wsi):
            #     print(
            #         f"Processing patient {patient_idx + 1}/{len(x_wsi)} with {len(patient_tiles)} tiles"
            #     )
            #     # Ensure correct shape (C, H, W) -> (1, C, H, W)
            #     processed_tiles = []
            #     for tile in patient_tiles:
            #         if tile.dim() == 3:  # (C, H, W)
            #             tile = tile.unsqueeze(0)  # add batch dimension -> (1, C, H, W)
            #         elif tile.dim() == 4 and tile.size(0) == 1:
            #             pass
            #         else:
            #             raise ValueError(f"Unexpected tile shape: {tile.shape}")
            #         processed_tiles.append(tile)

            #     # get embedding for this patient
            #     patient_embedding = self.encoder.get_wsi_embeddings(processed_tiles)
            #     # ensure the embedding is properly shaped on right device
            #     if not isinstance(patient_embedding, torch.Tensor):
            #         patient_embedding = torch.tensor(
            #             patient_embedding, dtype=torch.float32
            #         )
            #     if len(processed_tiles) > 0:
            #         patient_embedding = patient_embedding.to(processed_tiles[0].device)

            #     patient_embedding = self.net(
            #         patient_embedding
            #     )  # (1, 384) -> (1, embedding_dim_wsi)

            #     if patient_embedding.dim() > 1 and patient_embedding.size(0) == 1:
            #         patient_embedding = patient_embedding.squeeze(0)

            #     batch_embeddings.append(patient_embedding)

            # batch_embeddings = torch.stack(batch_embeddings)

            # return batch_embeddings  # shape: (batch_size, embedding_dim)

        else:
            return self.net(x_wsi)


class OmicNetwork(nn.Module):  # MLP for WSI tile-level embedding generation
    def __init__(
        self,
        embedding_dim,
        dropout=0.2,
        use_pretrained_vae=True,
        vae_checkpoint_path="checkpoint/checkpoint_2024-09-04-07-56-47/checkpoint_epoch_1500.pth",
    ):
        super(OmicNetwork, self).__init__()
        input_dim = 19962
        self.embedding_dim = embedding_dim
        self.use_pretrained_vae = use_pretrained_vae

        if use_pretrained_vae and vae_checkpoint_path:
            # Load pretrained VAE
            from generate_rnaseq_embeddings import BetaVAE

            self.vae_encoder = BetaVAE(
                input_dim=input_dim, latent_dim=256, intermediate_dim=512, beta=0.005
            )

            # Load pretrained weights
            checkpoint = torch.load(vae_checkpoint_path, map_location=device)
            new_state_dict = {
                key.replace("module.", ""): value
                for key, value in checkpoint["model_state_dict"].items()
            }

            self.vae_encoder.load_state_dict(new_state_dict)
            for param in self.vae_encoder.parameters():
                param.requires_grad = True

            self.projection = (
                nn.Linear(256, embedding_dim) if embedding_dim != 256 else nn.Identity()
            )

        else:

            self.net = nn.Sequential(
                # nn.Dropout(dropout),
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                # nn.Dropout(dropout),
                nn.Linear(256, embedding_dim),
            )

    def forward(self, x):
        # print("+++++++++++++ Input shape within omic network: ", x.shape)

        if self.use_pretrained_vae:
            mean, _ = self.vae_encoder.encode(x)
            return self.projection(mean)
        else:
            return self.net(x)


class MultimodalNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim_wsi,
        embedding_dim_omic,
        mode,
        fusion_type,
        mlp_layers,
        dropout=0.2,
        joint_embedding_type="weighted_avg",
        use_pretrained_omic=True,
        omic_checkpoint_path="checkpoints/checkpoint_2024-09-04-07-56-47/checkpoint_epoch_1500.pth",
    ):
        super(MultimodalNetwork, self).__init__()

        self.mode = mode  # wsi_omic, wsi or omic
        self.fusion_type = fusion_type

        # NOTE: Initial weights or final weights are choosen from the results from early fusion

        if joint_embedding_type == "weighted_avg_dynamic":
            self.omic_weight = nn.Parameter(torch.tensor(0.8))
            self.wsi_weight = nn.Parameter(torch.tensor(0.2))
        else:
            # NOTE: Default, even if concatenation because does not take that much memory
            self.omic_weight = 0.8
            self.wsi_weight = 0.2

        if self.mode != "wsi_omic":
            self.fusion_type = None
        print(f"Initializing with mode={mode}, fusion_type={fusion_type}")
        # if self.fusion_type is not None: # not unimodal

        if self.mode == "wsi_omic":
            self.wsi_net = WSINetwork(embedding_dim_wsi)
            self.omic_net = OmicNetwork(
                embedding_dim_omic,
                use_pretrained_vae=use_pretrained_omic,
                vae_checkpoint_path=omic_checkpoint_path,
            )

            # Don't create linear projection if non-weighted as not needed for concatenation
            if joint_embedding_type == "concatenate":

                embedding_dim = self.wsi_net.embedding_dim + self.omic_net.embedding_dim
                self.wsi_projection = nn.Linear(embedding_dim_wsi, embedding_dim_wsi)
                self.omic_projection = nn.Linear(embedding_dim_omic, embedding_dim_omic)
            else:
                embedding_dim = min(
                    self.wsi_net.embedding_dim, self.omic_net.embedding_dim
                )

                self.wsi_projection = nn.Linear(embedding_dim_wsi, embedding_dim)
                self.omic_projection = nn.Linear(embedding_dim_omic, embedding_dim)

        elif self.mode == "wsi":
            self.wsi_net = WSINetwork(embedding_dim_wsi)
            self.omic_net = None
            embedding_dim = self.wsi_net.embedding_dim

        elif self.mode == "omic":
            self.wsi_net = None
            self.omic_net = OmicNetwork(embedding_dim_omic)
            embedding_dim = self.omic_net.embedding_dim

        else:
            raise ValueError(f"Mode not recognized: {self.mode}")

        print(
            "Embedding dimension based on which the dimension of the input of the downstream MLP is set: ",
            embedding_dim,
        )

        # Downstream MLP for fused data (directly tied to the Cox loss)

        # ADD: change intermed dimension or add more layers

        layers = get_mlp_layers(embedding_dim, mlp_layers)
        self.fused_mlp = self._create_mlp(layers, dropout=dropout)

        print("##############  fused MLP Summary  ##################")
        print_model_summary(self.fused_mlp)

        self.stored_omic_embedding = None

    def forward(self, opt, tcga_id, x_wsi=None, x_omic=None):

        ################### DEBUG ########################
        # print("x_wsi: ", x_wsi) # contains float values and not the image files here
        # print("x_omic: ", x_omic)

        # print(f"=== FORWARD DEBUG  ===")
        # print(f"Batch size (tcga_id length): {len(tcga_id)}")
        # print(f"x_wsi type: {type(x_wsi)}")
        # print(f"x_wsi length (should equal batch size): {len(x_wsi)}")

        # if x_wsi and len(x_wsi) > 0:
        #     print(f"First patient's tiles: {len(x_wsi[0])} tiles")
        #     print(f"First tile shape: {x_wsi[0][0].shape}")

        # print(f"x_omic shape: {x_omic.shape}")
        # print(f"=== END DEBUG ===")

        ################### END DEBUG ########################

        start_time = time.time()
        # print("fusion type: ", self.fusion_type)
        if self.fusion_type == "joint":

            wsi_embedding = self.wsi_net(x_wsi)
            omic_embedding = self.omic_net(x_omic)
            # print("wsi_embedding.shape: ", wsi_embedding.shape)
            # print("omic_embedding.shape: ", omic_embedding.shape)
        step1_time = time.time()
        # set_trace()
        if self.fusion_type == "joint_omic":
            wsi_embedding = pd.read_json(
                os.path.join(opt.input_wsi_embeddings_path, "WSI_embeddings.json")
            )  # read pre-generated embeddings from the pathology foundation model
            wsi_embedding = wsi_embedding[
                list(tcga_id)
            ]  # keep only the embeddings corresponding to the tcga_ids in the batch
            print("Shape of omic input before OmicNetwork", x_omic.shape)
            omic_embedding = self.omic_net(x_omic)
            print("wsi_embedding.shape: ", wsi_embedding.shape)
            print("omic_embedding.shape: ", omic_embedding.shape)

        elif self.fusion_type == "early":
            # wsi_embedding = self.wsi_encoder.get_wsi_embeddings(x_wsi)  # get from pretrained foundation models; x_wsi contain data from all tiles
            # omic_embedding = get_omic_embeddings(x_omic)  # get from a simple VAE based encoder
            # for early fusion, we can get the embeddings from lookup tables
            wsi_embedding = early_fusion_get_wsi_embeddings
            omic_embedding = early_fusion_get_omic_embeddings

            print("wsi_embedding.shape: ", wsi_embedding.shape)
            print("omic_embedding.shape: ", omic_embedding.shape)

        elif self.fusion_type is None:  # unimodal case
            print("This is a unimodal case")
            if self.mode == "wsi":
                wsi_embedding = self.wsi_encoder.get_wsi_embeddings(
                    x_wsi
                )  # x_wsi contain data from all tiles
                print(
                    "wsi_embedding.shape (should be [batch_size, embedding_dim]): ",
                    wsi_embedding.shape,
                )
            elif self.mode == "omic":
                if (
                    self.stored_omic_embedding is None and x_omic is not None
                ):  # to avoid calling this function for every forward pass
                    self.stored_omic_embedding = get_omic_embeddings(x_omic)
                    self.stored_omic_embedding = torch.tensor(
                        self.stored_omic_embedding, dtype=torch.float32
                    ).to(x_wsi[0].device)
                    print(
                        "omic_embedding.shape (should be [batch_size, embedding_dim]): ",
                        self.stored_omic_embedding.shape,
                    )
                omic_embedding = (
                    self.stored_omic_embedding
                )  # reuse the stored embeddings for early fusion

        print("input mode: ", self.mode)

        if self.mode == "wsi":
            combined_embedding = wsi_embedding
        elif self.mode == "omic":
            combined_embedding = omic_embedding
        elif self.mode == "wsi_omic" and (
            self.fusion_type == "joint_omic" or self.fusion_type == "joint"
        ):

            # set_trace()

            # WARNING: Do not re-create tensors from embeddings if they are already tensors, otherwise there may be a graph disconnect on backprop

            if not isinstance(wsi_embedding, torch.Tensor):
                wsi_embedding = torch.tensor(wsi_embedding).to(device)
            if not isinstance(omic_embedding, torch.Tensor):
                omic_embedding = torch.tensor(omic_embedding).to(device)

            # with the joint loss the embeddings must be the same dim

            wsi_projected = self.wsi_projection(wsi_embedding)
            omic_projected = self.omic_projection(omic_embedding)
            raw_wsi_embedding = wsi_projected
            raw_omic_embedding = omic_projected

            if (
                opt.joint_embedding == "weighted_avg"
                or opt.joint_embedding == "weighted_avg_dynamic"
            ):

                total_weight = self.omic_weight + self.wsi_weight
                normalized_rna = self.omic_weight / total_weight
                normalized_wsi = self.wsi_weight / total_weight
                combined_embedding = (
                    normalized_rna * omic_projected + normalized_wsi * wsi_projected
                )
                # print(
                #     f"Combined embedding shape: {combined_embedding.shape}; Omic embedding shape: {omic_projected.shape}; WSI embedding shape: {wsi_projected.shape}"
                # )

            elif opt.joint_embedding == "concatenate":

                combined_embedding = torch.cat((wsi_embedding, omic_embedding), dim=1)

            else:

                print("Defaulting to weighted non-dynamic joint embedding...")

                total_weight = self.omic_weight + self.wsi_weight
                normalized_rna = self.omic_weight / total_weight
                normalized_wsi = self.wsi_weight / total_weight
                combined_embedding = (
                    normalized_rna * omic_projected + normalized_wsi * wsi_projected
                )

        step2_time = time.time()

        # print("combined_embedding.shape: ", combined_embedding.shape)
        # use combined embedding with downstream MLP for getting the output that enters the loss function
        step3_time = time.time()
        output = self.fused_mlp(combined_embedding)
        # # Ensure output is properly shaped - should be (batch_size,) or (batch_size, 1)
        # if output.dim() > 1 and output.size(1) == 1:
        #     output = output.squeeze(1)  # Convert (batch_size, 1) to (batch_size,)
        # step2_time = time.time()
        print(
            f"(In MultimodalNetwork) Step 1: {step1_time - start_time:.4f}s, Step 2: {step2_time - step1_time:.4f}s, Step 3: {step3_time - step2_time:.4f}s"
        )

        return output, raw_wsi_embedding, raw_omic_embedding

    def forward_omic_only(self, x_omic):
        omic_embedding = self.omic_net(x_omic)
        output = self.fused_mlp(omic_embedding)
        return output

    def _create_mlp(self, layers, dropout=0.2):
        """Create MLP with specified layers"""

        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(dropout))

        return nn.Sequential(*modules)


def print_model_summary(model):
    if model is None:
        print("model is NoneType")
        return
    total_params = sum(p.numel() for p in model.parameters())
    # print("NOTE: these do not account for the memory required for storing the optimizer states and the activations")
    print(f"Total Parameters (million): {total_params / 1e6}")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Number of trainable params (million): {total_trainable_params / 1e6}")

    # memory_bytes = total_params * 4  # 4 bytes for a torch.float32 model parameter
    # memory_mb = memory_bytes / (1024 ** 2)
    # memory_gb = memory_bytes / 1e9
    # print(f"Estimated Memory (GB): {memory_gb}")
