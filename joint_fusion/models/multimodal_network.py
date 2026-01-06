import torch
import torch.nn as nn
import pandas as pd
import os
import time

from .wsi_network import WSINetwork
from .omic_network import OmicNetwork

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
        use_pretrained_omic=False,
        omic_checkpoint_path="checkpoints/checkpoint_2024-09-04-07-56-47/checkpoint_epoch_1500.pth",
    ):
        super(MultimodalNetwork, self).__init__()

        self.mode = mode  # wsi_omic, wsi or omic
        self.fusion_type = fusion_type

        # NOTE: Initial weights or final weights are choosen from the results from early fusion

        if joint_embedding_type == "weighted_avg_dynamic":
            self.omic_weight = nn.Parameter(torch.tensor(0.8))
            self.wsi_weight = nn.Parameter(torch.tensor(0.2))
        elif joint_embedding_type == "weighted_avg":
            # NOTE: Default, even if concatenation because does not take that much memory
            self.omic_weight = 0.8
            self.wsi_weight = 0.2

        else:  # no longer weighted avg so treat as 1
            self.omic_weight = 1
            self.wsi_weight = 1

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
            else:
                embedding_dim = min(
                    self.wsi_net.embedding_dim, self.omic_net.embedding_dim
                )

                self.wsi_projection = nn.Linear(embedding_dim_wsi, embedding_dim)
                self.omic_projection = nn.Linear(embedding_dim_omic, embedding_dim)

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

    def forward(self, config, tcga_id, x_wsi=None, x_omic=None):
        # print("x_wsi: ", x_wsi) # contains float values and not the image files here
        # print("x_omic: ", x_omic)

        start_time = time.time()
        # print("fusion type: ", self.fusion_type)
        if self.fusion_type == "joint":
            wsi_embedding = self.wsi_net(x_wsi)
            omic_embedding = self.omic_net(x_omic)
            # print("wsi_embedding.shape: ", wsi_embedding.shape)
            # print("omic_embedding.shape: ", omic_embedding.shape)
        step1_time = time.time()

        print("input mode: ", self.mode)

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
            config.model.joint_embedding == "weighted_avg"
            or config.model.joint_embedding == "weighted_avg_dynamic"
        ):

            total_weight = self.omic_weight + self.wsi_weight
            normalized_rna = self.omic_weight / total_weight
            normalized_wsi = self.wsi_weight / total_weight
            combined_embedding = (
                normalized_rna * omic_projected + normalized_wsi * wsi_projected
            )

        elif config.model.joint_embedding == "concatenate":

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

        print("combined_embedding.shape: ", combined_embedding.shape)
        # use combined embedding with downstream MLP for getting the output that enters the loss function
        step3_time = time.time()
        output = self.fused_mlp(combined_embedding)
        # Ensure output is properly shaped - should be (batch_size,) or (batch_size, 1)
        if output.dim() > 1 and output.size(1) == 1:
            output = output.squeeze(1)  # Convert (batch_size, 1) to (batch_size,)
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
