import torch
import torch.nn as nn
from .wsi_encoder import WSIEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WSINetwork(nn.Module):
    def __init__(self, config, embedding_dim):
        super(WSINetwork, self).__init__()
        self.embedding_dim = embedding_dim

        self.encoder = WSIEncoder(
            config.model.wsi_fm,
            config.model.pooling,
            pretrained=True,
            tile_batch_size=config.model.wsi_encoder_tile_batch_size,
        )
        encoder_dim = getattr(self.encoder, "embed_dim", 384)
        self.net = nn.Sequential(
            nn.Linear(
                encoder_dim, embedding_dim
            ),  # to match the embedding dimension to the vit output
            nn.ReLU(),
        )
        self.last_attention_weights = None

    def forward(self, x_wsi):
        result = self.encoder.get_wsi_embeddings(x_wsi)
        if isinstance(result, tuple):
            embeddings, attention_weights = result
            self.last_attention_weights = attention_weights.detach().cpu()
        else:
            embeddings = result
        embeddings = embeddings.to(self.encoder.device)
        return self.net(embeddings)

    # ---- chunked-saliency helpers (forward() above is unchanged) ----
    # By construction, for any tiles x,
    #     forward(x) == forward_from_tile_features(encode_tile_features(x)),
    # since encoder.get_wsi_embeddings == encode_tiles followed by pooling, and
    # self.net is applied identically in both. The split just exposes the
    # per-tile features E = encode_tile_features(x) as an intermediate.
    def encode_tile_features(self, x_wsi):
        r"""Per-tile FM features \(E\in\mathbb{R}^{T\times d_{enc}}\), pre-pooling."""
        return self.encoder.encode_tiles(x_wsi)

    def forward_from_tile_features(self, tile_features):
        """Pooling + projection (self.net) starting from per-tile features."""
        slide_embedding, attention_weights = self.encoder.pool_tile_features(
            tile_features
        )
        if attention_weights is not None:
            self.last_attention_weights = attention_weights.detach().cpu()
        slide_embedding = slide_embedding.to(self.encoder.device)
        return self.net(slide_embedding)
