import torch
import torch.nn as nn
from .wsi_encoder import WSIEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WSINetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(WSINetwork, self).__init__()
        self.embedding_dim = embedding_dim

        # using lunit_dino
        self.encoder = WSIEncoder(pretrained=True)
        self.net = nn.Sequential(
            nn.Linear(
                384, embedding_dim
            ),  # to match the embedding dimension to the vit output
            nn.ReLU(),
        )

    def forward(self, x_wsi):
        embeddings = self.encoder.get_wsi_embeddings(x_wsi)
        embeddings = embeddings.to(self.encoder.device)
        return self.net(embeddings)
