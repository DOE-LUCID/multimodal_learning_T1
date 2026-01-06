import torch
import torch.nn as nn

from .omic_encoder import BetaVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
