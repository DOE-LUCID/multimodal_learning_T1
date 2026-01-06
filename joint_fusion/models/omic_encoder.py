# Implement a MLP based VAE to create lower dimensional embeddings of the RNASeq data
# conda activate pytorch_py3p10 on laptop
# use 'wandb sweep sweep_config.yaml' followed by 'wandb agent <username>/<project_name>/<sweep_id>' [just follow the instructions after 'wandb sweep'] for HPO runs with wandb
# for running without HPO, simply use "python generate_rnaseq_embeddings.py"

# for running on a single GPU use: CUDA_VISIBLE_DEVICES=0 python generate_rnaseq_embeddings.py

import torch
from torch import nn
from torchsummary import summary
from torch.nn import functional as F
from pdb import set_trace


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print("Running on GPU")
else:
    print("Running on CPU")

# ######################### VAE HYPERPARAMETERS ############################
# lengths = mapping_df['rnaseq_data'].apply(lambda x: len(x))
# lengths_equal = lengths.all()
# if lengths_equal:
#     print("all tcga samples have the same number of genes")
# else:
#     print("WARNING: all tcga samples DO NOT have the same number of genes ")
#
# input_dim = lengths.iloc[0]  # number of genes (~ 20K)
# intermediate_dim = 128
# lr = 1e-4
# latent_dim = 64  # get from the input opt
# beta = 0.005  # 0.01   # 0: equivalent to standard autoencoder; 1: equivalent to standard VAE

# train_batch_size = 128
# val_batch_size = 8
# test_batch_size = 8
# num_epochs = 200000


class BetaVAE(nn.Module):
    def __init__(
        self,
        input_dim=None,
        latent_dim=None,
        intermediate_dim=None,
        beta=None,
        config=None,
    ):
        super(BetaVAE, self).__init__()
        if (
            config
        ):  # during training with HPO using wandb (read parameters from the sweep_config.yaml file)
            self.input_dim = config.input_dim
            self.latent_dim = config.latent_dim
            self.intermediate_dim = config.intermediate_dim
            self.lr = config.learning_rate
        else:
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.intermediate_dim = intermediate_dim
            self.beta = beta

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.LeakyReLU(),
            nn.Linear(intermediate_dim, latent_dim * 2),  # *2 for mean and log_var
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.LeakyReLU(),
            nn.Linear(intermediate_dim, input_dim),
            # nn.Sigmoid()  # assuming data is normalized between 0 and 1
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, log_var = torch.chunk(x, 2, dim=-1)  # log_var = log(std**2)
        return mean, log_var

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)  # to ensure the variance is > 0
        eps = torch.randn_like(std).to(device)
        z = mean + eps * std
        # set_trace()
        return self.decode(z), mean, log_var


def loss_function(recon_x, x, mean, log_var):
    reconstruction_loss = F.mse_loss(recon_x, x, reduction="sum")
    kl_div_loss = -0.5 * torch.sum(
        1 + log_var - mean.pow(2) - log_var.exp()
    )  # See Geron Eq 17-4
    return reconstruction_loss, kl_div_loss
