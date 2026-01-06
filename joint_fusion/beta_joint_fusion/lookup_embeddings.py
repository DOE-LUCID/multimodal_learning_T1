# script to lookup embeddings for early fusion
# the embeddings are pre-written out before training the downstream MLP
import pandas as pd
from pdb import set_trace


def early_fusion_get_wsi_embeddings():
    embeddings = pd.read_json('WSI_embeddings.json')
    return embeddings


def early_fusion_get_omic_embeddings():
    embeddings = pd.read_json('rnaseq_embeddings.json')
    return embeddings

if __name__ == "__main__":
    embeddings_wsi = early_fusion_get_wsi_embeddings()
    embeddings_rnaseq = early_fusion_get_omic_embeddings()
    set_trace()