# code for training VAE and generating embeddings for bulk RNASeq data for early fusion
# Implement a MLP based VAE to create lower dimensional embeddings of the RNASeq data
# run directly for generating embeddings for early fusion
# otherwise, functions used by models.py for joint fusion

# conda activate pytorch_py3p10 on laptop
# use 'wandb sweep sweep_config.yaml' followed by 'wandb agent <username>/<project_name>/<sweep_id>' [just follow the instructions after 'wandb sweep'] for HPO runs with wandb
# for running without HPO, simply use "python generate_rnaseq_embeddings.py"

# for running on a single GPU use: CUDA_VISIBLE_DEVICES=0 python generate_rnaseq_embeddings.py

import torch
from torch import nn, optim
import copy
import os
import argparse
import joblib
import json
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchsummary import summary
from torch.nn import functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
from datasets import CustomDataset
import wandb
from pdb import set_trace

if __name__ == "__main__": # for early fusion
    from train_test import create_data_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if device.type == 'cuda':
    print("Running on GPU")
else:
    print("Running on CPU")

class BetaVAE(nn.Module):
    def __init__(self, input_dim=None, latent_dim=None, intermediate_dim=None, beta=None, config=None):
        super(BetaVAE, self).__init__()
        if config:  # during training with HPO using wandb (read parameters from the sweep_config.yaml file)
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
            nn.Linear(intermediate_dim, latent_dim * 2)  # *2 for mean and log_var
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.LeakyReLU(),
            nn.Linear(intermediate_dim, input_dim)
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
    reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())  # See Geron Eq 17-4
    return reconstruction_loss, kl_div_loss


# function to take a trained VAE and generate embeddings for a new rnaseq dataset
# essentially, it just passes the input through the trained encoder network
# for early fusion, these embeddings should be loaded only once (before training of the downstream MLP starts)
def get_omic_embeddings(x_omic, latest_checkpoint, checkpoint_dir=None):
    print("In get_omic_embeddings()")
    # work directly on the tensor on the device
    # x_omic = torch.log1p(x_omic) # not required as already done in the data loader
    input_dim = 19962 # remove this hard-coded value

    model = BetaVAE(input_dim=input_dim,
                    latent_dim=opt.latent_dim,
                    intermediate_dim=opt.intermediate_dim,
                    beta=opt.beta)

    # checkpoints = [file for file in os.listdir(checkpoint_dir) if file.endswith('.pth')]
    # latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    # latest_checkpoint = 'checkpoint_epoch_80000.pth'

    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # for some reason 'module' gets added to the names of the model parameters, so adjusting the state dictionary to remove 'module.' prefix
    new_state_dict = {key.replace("module.", ""): value for key, value in checkpoint['model_state_dict'].items()}
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(new_state_dict)
    print(f"loaded checkpoint from {checkpoint_path}")
    model.to(device)
    model.eval()
    embeddings = []
    # forward pass through the encoder to obtain the embeddings
    with torch.no_grad():
        for data in DataLoader(TensorDataset(torch.tensor(x_omic, dtype=torch.float)), batch_size=1):
            input = data[0].to(device)
            mean, _ = model.encode(input)
            embeddings.append(mean.cpu().numpy())
    # concatenate all batch embeddings into a single numpy array
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def main(opt):
    restart_training = False
    wandb.init(project="rnaseq_vae", entity="tnnandi")
    config = wandb.config
    config = None  # remove this when using the sweep_config.yaml file

    h5_file = 'mapping_data.h5'
    train_loader, _, _ = create_data_loaders(opt, h5_file)
    # # determine input dimension from the first batch of omic data
    # for _, _, _, _, x_omic in train_loader:
    #     input_dim = x_omic.shape[1]  # assuming x_omic is of shape (batch_size, num_genes)
    #     break
    input_dim = 19962 # remove this hard-coding

    reconstruction_loss, kl_divergence_loss, val_loss = None, None, None

    # if torch.cuda.device_count() > 1:
    #     print(f"using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)

    # create multiple folds from the training data
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold CV
    dataset = train_loader.dataset  # use the training dataset for CV

    total_samples = len(dataset)
    print(f"total training data size: {total_samples} samples")

    # train models over all folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{kf.get_n_splits()}")

        # create training subset from the fold
        print("create training and validation subsets from the folds")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        # create new DataLoader for this fold
        train_loader_fold = DataLoader(train_subset,
                                       batch_size=opt.batch_size,
                                       shuffle=True,)
        val_loader_fold = DataLoader(val_subset,
                                       batch_size=opt.batch_size,
                                       shuffle=False,)

        # number of samples and batches for the current fold
        num_samples = len(train_subset)
        num_batches = len(train_loader_fold)
        num_samples_val = len(val_subset)
        print(f"Number of samples in training fold {fold}: {num_samples}, in validation fold: {num_samples_val}")
        print(f"Number of batches in training fold {fold}: {num_batches}")

        # initialize model, optimizer, scheduler for each fold
        # **reinitialize model for each fold**
        model = BetaVAE(input_dim=input_dim,
                        latent_dim=opt.latent_dim,
                        intermediate_dim=opt.intermediate_dim,
                        beta=opt.beta)

        if torch.cuda.device_count() > 1:
            print(f"using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        # scheduler = CosineAnnealingLR(optimizer, T_max=opt.num_epochs, eta_min=opt.lr_min)
        scheduler = ExponentialLR(optimizer, gamma=0.999)

        # initialize and fit the scaler incrementally on training data
        print("fitting scaler")
        scaler = StandardScaler()
        for batch_idx, (tcga_id, _, _, _, x_omic) in enumerate(train_loader_fold):
            x_omic = x_omic.cpu().numpy()
            scaler.partial_fit(x_omic)

        # save the scaler for the current fold
        scaler_path = os.path.join(checkpoint_dir, f'scaler_fold_{fold}.save')
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved for fold {fold} at {scaler_path}")

        for epoch in range(opt.num_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{opt.num_epochs}, Fold {fold}, LR: {current_lr:.10f}")
            model.train()
            train_loss = 0.0
            for batch_idx, (tcga_id, _, _, _, x_omic) in enumerate(train_loader_fold):
                print("batch_index: ", batch_idx)
                optimizer.zero_grad()
                x_omic = x_omic.to(device)
                recon_batch, mean, log_var = model(x_omic)
                reconstruction_loss, kl_divergence_loss = loss_function(recon_batch, x_omic, mean, log_var)
                loss = reconstruction_loss + opt.beta * kl_divergence_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # print training loss, calculate validation loss, and save trained model every 100 epochs
            if epoch % 100 == 0 and epoch > 0:
                train_loss /= len(train_loader_fold.dataset)
                print(f"Training loss at epoch {epoch}: {train_loss}")

                # validation step (in eval mode)
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_idx, (tcga_id, _, _, _, x_omic) in enumerate(val_loader_fold):
                        # x_omic = x_omic.to(device)
                        x_omic = x_omic.cpu().numpy()
                        x_omic = scaler.transform(x_omic)
                        x_omic = torch.tensor(x_omic, dtype=torch.float32).to(device)
                        recon_batch, mean, log_var = model(x_omic)
                        reconstruction_loss, kl_divergence_loss = loss_function(recon_batch, x_omic, mean, log_var)
                        loss = reconstruction_loss + opt.beta * kl_divergence_loss
                        val_loss += loss.item()

                val_loss /= len(val_loader_fold.dataset)
                print(f"Validation loss at epoch {epoch}: {val_loss}")

                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_fold_{fold}_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch}, fold {fold}, to {checkpoint_path}")

            # track losses using weights and biases
            wandb.log({"epoch": epoch,
                       "fold": fold,
                       "LR": current_lr,
                       "train_loss": train_loss,
                       "val_loss": val_loss,
                       "reconstruction_loss": reconstruction_loss.item() if reconstruction_loss is not None else float('nan'),
                       "kl_div_loss": kl_divergence_loss.item() if kl_divergence_loss is not None else float('nan'),})

            scheduler.step()

    wandb.finish()


# for early fusion training, and inference (generate embeddings from trained model)
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training = False
    inference = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wsi_path', type=str,
                        # default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/TCGA_WSI/batch_corrected/processed_svs/tiles/256px_9.9x/combined_tiles/', # on laptop
                        default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles/tiles/256px_9.9x/combined_tiles/',
                        help='Path to input WSI tiles',
                        )
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training (overall)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='Batch size for validation data (using all samples for better Cox loss calculation)')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--input_size_wsi', type=int, default=256, help="input_size for path images")
    parser.add_argument('--input_mode', type=str, default="wsi_omic", help="wsi, omic, wsi_omic")
    parser.add_argument('--intermediate_dim', type=int, default=512, help='Dimension of intermediate layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (this is the max LR for the scheduler)')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum Learning rate')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space')
    parser.add_argument('--beta', type=float, default=0.005, help='Beta parameter for VAE')
    parser.add_argument('--num_epochs', type=int, default=4000, help='Number of epochs for training')

    opt = parser.parse_args()

    if training:
        current_time = datetime.now()
        checkpoint_dir = "checkpoint_" + current_time.strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(checkpoint_dir, exist_ok=True)

        main(opt)

    elif inference:
        # checkpoint_dir = "checkpoint_2024-09-03-02-17-37"
        # checkpoint_dir = "checkpoint_2024-09-04-07-56-47"  #
        # checkpoint_dir = "checkpoint_2024-10-18-06-49-51"
        checkpoint_dir = "checkpoint_2024-10-18-21-57-09"
        fold_id = 0
        latest_checkpoint_epoch = 2800# 1900 #1500
        latest_checkpoint = f'checkpoint_fold_{fold_id}_epoch_{latest_checkpoint_epoch}.pth'

        # load the saved scaler for normalizing x_omic
        scaler = joblib.load(os.path.join(checkpoint_dir, f'scaler_fold_{fold_id}.save'))
        mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
        scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

        h5_file = 'mapping_data.h5'
        train_loader, val_loader, test_loader = create_data_loaders(opt, h5_file)

        # force batch size of 1 for each loader for inference
        train_loader = torch.utils.data.DataLoader(dataset=train_loader.dataset,
                                                   batch_size=1)
        val_loader   = torch.utils.data.DataLoader(dataset=val_loader.dataset,
                                                   batch_size=1)
        test_loader  = torch.utils.data.DataLoader(dataset=test_loader.dataset,
                                                   batch_size=1)

        # initialize a dictionary to store embeddings for each split
        split_patient_embeddings = {
            'train': {},
            'val': {},
            'test': {}
        }

        # choose the datasets for which inference needs to be done
        loaders = [train_loader, val_loader, test_loader]
        split_names = ['train', 'val', 'test'] # make it consistent with the list in loaders above

        # loop over each loader and process the data
        for split_name, loader in zip(split_names, loaders):
            print(f"Processing {split_name} split...")
            for batch_idx, (tcga_id, days_to_event, event_occurred, x_wsi, x_omic) in enumerate(loader):
                tcga_id = tcga_id[0]  # assuming tcga_id is a batch of size 1
                print(f"TCGA ID: {tcga_id}, batch_idx: {batch_idx}, out of {len(loader)} for split {split_name}")
                x_omic = x_omic.to(device)
                x_omic = (x_omic - mean) / scale
                # get the omic embeddings using the checkpoint
                embeddings = get_omic_embeddings(x_omic, latest_checkpoint, checkpoint_dir=checkpoint_dir)
                embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

                # store the embeddings in the corresponding split dictionary
                split_patient_embeddings[split_name][tcga_id] = embeddings_list

        # save the embeddings for each split to separate json files
        for split_name in split_names:
            # filename = f"./rnaseq_embeddings_{split_name}_{latest_checkpoint_epoch}.json"
            filename = os.path.join(checkpoint_dir, f"rnaseq_embeddings_{split_name}_{checkpoint_dir}_fold_{fold_id}_epoch_{latest_checkpoint_epoch}.json")
            with open(filename, 'w') as file:
                json.dump(split_patient_embeddings[split_name], file)



# set_trace()
