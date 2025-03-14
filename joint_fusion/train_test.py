import random
from tqdm import tqdm
import numpy as np
import torch
import os
import wandb
import time
import argparse
import json
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader, Subset
# import torch.optim.lr_scheduler as lr_scheduler
from torchsummary import summary
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from utils import mixed_collate
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from sklearn.preprocessing import StandardScaler

from datasets import CustomDataset, HDF5Dataset
from models import MultimodalNetwork, OmicNetwork, print_model_summary
from sklearn.model_selection import KFold
from generate_wsi_embeddings import CustomDatasetWSI

import h5py

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

if __name__ == "__main__":
    os.environ[
        "CUDA_VISIBLE_DEVICES"] = "0"  # force to use only one GPU to avoid any issues with the Cox PH loss function (that requires data for all at-risk samples)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pdb
import pickle
import os
from pdb import set_trace
from captum.attr import IntegratedGradients, Saliency

import torchviz

torch.autograd.set_detect_anomaly(True)

current_time = datetime.now().strftime("%y_%m_%d_%H_%M")


class CoxLoss(nn.Module):
    def __init__(self):
        super(CoxLoss, self).__init__()

    def forward(self, log_risks, times, censor):
        """
        :param log_risks: predictions from the NN
        :param times: observed survival times (i.e. times to death) for the batch
        :param censor: censor data, event (death) indicators (1/0)
        :return: Cox loss (scalar)
        """
        sorted_times, sorted_indices = torch.sort(times, descending=True)
        sorted_log_risks = log_risks[sorted_indices]
        sorted_censor = censor[sorted_indices]

        # precompute for using within the inner sum of term 2 in Cox loss
        exp_sorted_log_risks = torch.exp(sorted_log_risks)

        # initialize all samples to be at-risk (will update it below)
        at_risk_mask = torch.ones_like(sorted_times, dtype=torch.bool)

        losses = []
        for time_index in range(len(sorted_times)):
            # include only the uncensored samples (i.e., for whom the event has happened)
            if sorted_censor[time_index] == 1:
                at_risk_mask = torch.arange(
                    len(sorted_times)) <= time_index  # less than, as sorted_times is in descending order
                at_risk_mask = at_risk_mask.to(device)
                at_risk_sum = torch.sum(exp_sorted_log_risks[  # 2nd term on the RHS
                                            at_risk_mask])  # all are at-risk for the first sample (after arranged in descending order)
                loss = sorted_log_risks[time_index] - torch.log(at_risk_sum + 1e-15)
                losses.append(loss)

            # at_risk_mask[time_index] = False # the i'th sample is no more in the risk-set as the event has already occurred for it

        # if no uncensored samples are in the mini-batch return 0
        if not losses:
            return torch.tensor(0.0, requires_grad=True)

        cox_loss = -torch.mean(torch.stack(losses))
        return cox_loss


def create_data_loaders(opt, h5_file):
    train_loader = torch.utils.data.DataLoader(
        dataset=HDF5Dataset(opt, h5_file, split='train', mode=opt.input_mode, train_val_test="train"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,  # 8,
        # prefetch_factor=2,
        pin_memory=True
    )

    validation_loader = torch.utils.data.DataLoader(
        dataset=HDF5Dataset(opt, h5_file, split='val', mode=opt.input_mode, train_val_test="val"),
        batch_size=opt.val_batch_size,
        shuffle=True,
        num_workers=0,  # 8,
        # prefetch_factor=2,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=HDF5Dataset(opt, h5_file, split='test', mode=opt.input_mode, train_val_test="test"),
        batch_size=opt.test_batch_size,
        shuffle=True,
        num_workers=0,  # 1, #4,
        # prefetch_factor=2,
        pin_memory=True
    )

    return train_loader, validation_loader, test_loader


def train_nn(opt, h5_file, device):
    wandb.init(project="multimodal_survival_analysis", entity='tnnandi')
    config = wandb.config
    current_time = datetime.now()
    checkpoint_dir = "checkpoint_" + current_time.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader, _, test_loader = create_data_loaders(opt, h5_file)
    # create multiple folds from the training data
    kf = KFold(n_splits=opt.n_folds, shuffle=True, random_state=6)  # k-fold CV where k = opt.n_folds
    dataset = train_loader.dataset  # use only the training dataset for CV during training [both the validation and test splits can be used for testing]

    total_samples = len(dataset)
    print(f"total training data size: {total_samples} samples")

    # train models for each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{kf.get_n_splits()}")

        # create train and validation subsets (from the training data itself)
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # create data loaders for this fold
        train_loader_fold = DataLoader(train_subset,
                                       batch_size=opt.batch_size,
                                       num_workers=4,
                                       pin_memory=True,
                                       shuffle=True,
                                       drop_last=True)
        val_loader_fold = DataLoader(val_subset,
                                     batch_size=opt.val_batch_size,
                                     shuffle=False)

        # initialize model, optimizer, and scheduler for this fold
        model = MultimodalNetwork(embedding_dim_wsi=opt.embedding_dim_wsi,
                                  embedding_dim_omic=opt.embedding_dim_omic,
                                  mode=opt.input_mode,
                                  fusion_type=opt.fusion_type)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        cox_loss = CoxLoss()

        # initialize and fit the scaler incrementally on training data
        print("fitting scaler [train_test.py]")
        scaler = StandardScaler()
        for batch_idx, (tcga_id, _, _, _, x_omic) in enumerate(train_loader_fold):
            print(f"fitting batch index {batch_idx} of {len(train_loader_fold)} batches")
            x_omic = x_omic.cpu().numpy()
            scaler.partial_fit(x_omic)

        # save the scaler for the current fold
        scaler_path = os.path.join(checkpoint_dir, f'scaler_fold_{fold}.save')
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved for fold {fold} at {scaler_path}")

        # training loop
        for epoch in tqdm(range(0, opt.num_epochs)):
            print(f"**********  Fold {fold} out of {opt.n_folds},  Epoch: {epoch + 1} out of {opt.num_epochs}")
            start_train_time = time.time()
            model.train()
            # # added to reduce memory requirement
            # if isinstance(model, torch.nn.DataParallel):
            #     model.module.gradient_checkpointing_enable()
            # else:
            #     model.gradient_checkpointing_enable()
            loss_epoch = 0

            # log the learning rate at the start of the epoch
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"LR": current_lr}, step=epoch)

            # model training in batches for the train dataloader for the current fold
            for batch_idx, (tcga_id, days_to_event, event_occurred, x_wsi, x_omic) in enumerate(train_loader_fold):
                # x_wsi is a list of tensors (one tensor for each tile)
                print(f"Total training samples in fold: {len(train_loader_fold.dataset)}")
                print(f"Batch size: {opt.batch_size}")
                print(f"Batch index: {batch_idx + 1} out of {np.ceil(len(train_loader_fold.dataset) / opt.batch_size)}")
                x_wsi = [x.to(device) for x in x_wsi]
                x_omic = x_omic.to(device)
                days_to_event = days_to_event.to(device)
                # days_to_last_followup = days_to_last_followup.to(device)
                event_occurred = event_occurred.to(device)
                print("Days to event: ", days_to_event)
                print("event occurred: ", event_occurred)

                optimizer.zero_grad()

                if opt.use_mixed_precision:
                    with autocast():  # should wrap only the forward pass including the loss calculation
                        predictions = model(x_wsi=x_wsi,  # list of tensors (one for each tile)
                                            x_omic=x_omic)
                        loss = cox_loss(predictions.squeeze(),
                                        days_to_death,
                                        event_occurred)
                        print("\n loss: ", loss.data.item())
                        loss_epoch += loss.data.item()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    print(" Not using mixed precision")
                    # model for survival outcome (uses Cox PH partial log likelihood as the loss function)
                    # the model output should be considered as beta*X to be used in the Cox loss function

                    start_time = time.time()
                    predictions = model(opt,
                                        tcga_id,
                                        x_wsi=x_wsi,  # list of tensors (one for each tile)
                                        x_omic=x_omic,
                                        )
                    # print(f"predictions: {predictions} from train_test.py")
                    step1_time = time.time()
                    loss = cox_loss(predictions.squeeze(),
                                    # predictions are not survival outcomes, rather log-risk scores beta*X
                                    days_to_event,
                                    event_occurred)  # Cox partial likelihood loss for survival outcome prediction
                    print("\n loss (train): ", loss.data.item())
                    step2_time = time.time()
                    loss_epoch += loss.data.item()  # * len(tcga_id)  # multiplying loss by batch size for accurate epoch averaging
                    # backpropagate loss through the entire model arch upto the inputs
                    loss.backward(
                        retain_graph=True if epoch == 0 and batch_idx == 0 else False)  # tensors retained to allow backpropagation for torchhviz (for visualizing the graph)
                    optimizer.step()
                    torch.cuda.empty_cache()

                    step3_time = time.time()
                    print(
                        f"(in train_nn) Step 1: {step1_time - start_time:.4f}s, Step 2: {step2_time - step1_time:.4f}s, Step 3: {step3_time - step2_time:.4f}s")

                    # if epoch == 0 and batch_idx == 0:
                    #     # Note: graphviz is fine for small graphs, but for large graphs it becomes cumbersome so instead opting TensorBoard for the latter
                    #     # graph = torchviz.make_dot(loss, params=dict(model.named_parameters()))
                    #     # file_path = os.path.abspath("data_flow_graph")
                    #     # graph.render(file_path, format="png")
                    #     # print(f"Graph saved as {file_path}.png. You can open it manually.")
                    #
                    #     writer.add_graph(model_to_log, [x_wsi[0], x_omic])
                    #     print(f"Computation graph saved to TensorBoard logs at {log_dir}")

            train_loss = loss_epoch / len(train_loader_fold.dataset)  # average training loss per sample for the epoch
            wandb.log({"Loss/train": train_loss}, step=epoch)
            scheduler.step()  # step scheduler after each epoch
            print("\n train loss over epoch: ", train_loss)
            end_train_time = time.time()
            train_duration = end_train_time - start_train_time
            wandb.log({"Time/train": train_duration}, step=epoch)

            # calculate validation loss and calculate CI using validation data for this fold, and save model every 50 epochs
            if epoch % 50 == 0 and epoch > 0:
                train_loss /= len(train_loader_fold.dataset)
                print(f"Training loss at epoch {epoch}: {train_loss}")

                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_fold_{fold}_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch}, fold {fold}, to {checkpoint_path}")

                start_val_time = time.time()

                # get predictions on the validation dataset
                model.eval()
                val_loss_epoch = 0.0
                all_predictions = []
                all_times = []
                all_events = []

                with torch.no_grad():  # inference on the validation data
                    for batch_idx, (tcga_id, days_to_event, event_occurred, x_wsi, x_omic) in enumerate(
                            val_loader_fold):
                        # x_wsi is a list of tensors (one tensor for each tile)
                        print(f"Batch size: {len(val_loader_fold.dataset)}")
                        print(
                            f"Validation Batch index: {batch_idx + 1} out of {np.ceil(len(val_loader_fold.dataset) / opt.val_batch_size)}")
                        x_wsi = [x.to(device) for x in x_wsi]
                        x_omic = x_omic.to(device)
                        days_to_event = days_to_event.to(device)
                        event_occurred = event_occurred.to(device)
                        print("Days to event: ", days_to_event)
                        print("event occurred: ", event_occurred)
                        outputs = model(opt,
                                        tcga_id,
                                        x_wsi=x_wsi,  # list of tensors (one for each tile)
                                        x_omic=x_omic,
                                        )
                        loss = cox_loss(outputs.squeeze(),
                                        # predictions are not survival outcomes, rather log-risk scores beta*X
                                        days_to_event,
                                        event_occurred)  # Cox partial likelihood loss for survival outcome prediction
                        print("\n loss (validation): ", loss.data.item())
                        val_loss_epoch += loss.data.item() * len(tcga_id)
                        all_predictions.append(outputs.squeeze())
                        all_times.append(days_to_event)
                        all_events.append(event_occurred)

                        # model_path = os.path.join("./saved_models_4sep", f"model_epoch_{epoch}.pt") # change the dir name to use date time
                        # torch.save(model.state_dict(), model_path)
                        # print(f"saved model checkpoint at epoch {epoch} to {model_path}")

                    val_loss = val_loss_epoch / len(val_loader_fold.dataset)
                    wandb.log({"Loss/validation": val_loss}, step=epoch)

                    all_predictions = torch.cat(all_predictions)
                    all_times = torch.cat(all_times)
                    all_events = torch.cat(all_events)

                    # convert to numpy arrays for CI calculation
                    all_predictions_np = all_predictions.cpu().numpy()
                    all_times_np = all_times.cpu().numpy()
                    all_events_np = all_events.cpu().numpy()

                    c_index = concordance_index_censored(all_events_np.astype(bool), all_times_np, all_predictions_np)
                    print(f"Validation loss: {val_loss}, CI: {c_index[0]}")
                    wandb.log({"CI/validation": c_index[0]}, step=epoch)

                    # # # model_path = os.path.join("./saved_models_4sep", f"model_epoch_{epoch}.pt")
                    # save_dir = f"./saved_models_{current_time}"
                    # os.makedirs(save_dir, exist_ok=True)
                    model_path = os.path.join(checkpoint_dir, f"model_fold_{fold}_epoch_{epoch}.pt")
                    torch.save(model.state_dict(), model_path)
                    print(f"saved model checkpoint for fold {fold} at epoch {epoch} to {model_path}")

                    end_val_time = time.time()
                    val_duration = end_val_time - start_val_time
                    wandb.log({"Time/validation": val_duration}, step=epoch)

        return model, optimizer


def print_total_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")
        total_params += param.numel()
    print(f"Total parameters: {total_params / 1e6} million")
    print("Number of trainable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


def denormalize_image(image, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    image = (image * std) + mean
    image = np.clip(image, 0, 1)
    return image


def plot_saliency_maps(saliencies, x_wsi, tcga_id, patch_id, output_dir):
    # get the first image and saliency map
    saliency = saliencies[0].squeeze().cpu().numpy()
    # image = x_wsi[0].squeeze().permute(1, 2, 0).cpu().numpy()  # convert image to HxWxC and move to cpu
    image = x_wsi[0].squeeze().permute(1, 2, 0).detach().cpu().numpy()
    # denormalize the image based on normalization factors used during transformations of the test images
    mean = [0.70322989, 0.53606487, 0.66096631]
    std = [0.21716536, 0.26081574, 0.20723464]
    image = denormalize_image(image, mean, std)

    # normalize the saliency map to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("original patch")

    # overlay saliency map on the image
    ax = plt.subplot(1, 3, 2)
    img = ax.imshow(image)
    saliency_overlay = ax.imshow(saliency, cmap='hot', alpha=0.5)
    # saliency_overlay = ax.imshow(np.clip(saliency, 0.8, 1), cmap='hot', alpha=0.5)
    cbar = plt.colorbar(saliency_overlay, ax=ax)
    cbar.set_label('saliency value', rotation=270, labelpad=15)

    plt.title("saliency map overlay")

    # plot the saliency map alone
    plt.subplot(1, 3, 3)
    plt.imshow(saliency, cmap='hot')
    plt.colorbar(label='saliency value')
    plt.title("saliency map only")

    save_path = os.path.join(output_dir, f'saliency_overlay_{tcga_id[0]}_{patch_id}.png')
    plt.savefig(save_path)
    print(f"saved saliency overlay to {save_path}")

    plt.close()


def calc_integrated_gradients(model, opt, tcga_id, x_omic, x_wsi, baseline=None, steps=10):
    # baseline = None
    # baseline.shape
    # torch.Size([1, 19962])
    # set_trace()
    if baseline is None:
        print("** using trivial zero baseline for IG **")
        baseline = torch.zeros_like(x_omic).to(x_omic.device)
    else:
        print("** Using mean gene expression values over training samples as the baseline for IG **")
        baseline = torch.from_numpy(baseline).to(x_omic.device)
        baseline = baseline.float()
        baseline = baseline * torch.ones_like(x_omic).to(x_omic.device)
    # set_trace()
    print(f"CALCULATING INTEGRATED GRADIENTS OVER {steps} steps")
    scaled_inputs = [baseline + (float(i) / steps) * (x_omic - baseline) for i in range(steps + 1)]
    gradients = []
    steps_index = 0
    for scaled_input in scaled_inputs:
        print("steps_index: ", steps_index)
        scaled_input.requires_grad = True
        with torch.enable_grad():
            output = model(
                opt,
                tcga_id,
                x_wsi=x_wsi,  # list of tensors (one for each tile)
                x_omic=scaled_input,
            )
            print("output: ", output)
            output = output.sum()
            output.backward()
        gradients.append(scaled_input.grad.detach().cpu().numpy())
        steps_index += 1
    # set_trace()
    avg_gradients = np.mean(gradients[:-1], axis=0)
    integrated_grads = (x_omic.detach().cpu().numpy() - baseline.detach().cpu().numpy()) * avg_gradients

    return integrated_grads


def test_and_interpret(opt, model, test_loader, device, baseline=None):
    model.eval()
    test_loss_epoch = 0.0
    all_tcga_ids = []
    all_predictions = []
    all_times = []
    all_events = []

    # create the directory to save saliency maps if it doesn't exist
    output_dir = "./saliency_maps_6sep"
    os.makedirs(output_dir, exist_ok=True)

    output_dir_IG = "./IG_6sep"
    os.makedirs(output_dir_IG, exist_ok=True)

    # for training, only the last transformer block (block 11) in the WSI encoder was kept trainable
    # see WSIEncoder class in generate_Wsi_embeddings.py

    # Get the CI and the KM plots for the test set
    excluded_ids = ['TCGA-05-4395', 'TCGA-86-8281']  # contains anomalous time to event and censoring data
    # remove these ids during the input json/h5 file creation
    with torch.no_grad():
        for batch_idx, (tcga_id, days_to_event, event_occurred, x_wsi, x_omic) in enumerate(test_loader):
            if tcga_id[0] in excluded_ids:
                print(f"Skipping TCGA ID: {tcga_id}")
                continue

            x_wsi = [x.to(device) for x in x_wsi]
            x_omic = x_omic.to(device)
            days_to_event = days_to_event.to(device)
            event_occurred = event_occurred.to(device)

            # enable gradients only after data loading
            x_wsi = [x.requires_grad_() for x in x_wsi]

            print(f"Batch size: {len(test_loader.dataset)}")
            print(f"Test Batch index: {batch_idx + 1} out of {np.ceil(len(test_loader.dataset) / opt.test_batch_size)}")
            print("TCGA ID: ", tcga_id)
            print("Days to event: ", days_to_event)
            print("event occurred: ", event_occurred)

            if opt.calc_saliency_maps is False:
                outputs = model(
                    opt,
                    tcga_id,
                    x_wsi=x_wsi,  # list of tensors (one for each tile)
                    x_omic=x_omic,
                )

            if opt.calc_saliency_maps is True:
                # perform the forward pass without torch.no_grad() to allow gradient computation
                with torch.enable_grad():
                    outputs = model(
                        opt,
                        tcga_id,
                        x_wsi=x_wsi,  # list of tensors (one for each tile)
                        x_omic=x_omic,
                    )

                    # Check and print memory usage after each batch
                    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  # in GB
                    reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)  # in GB

                    print(f"After batch {batch_idx + 1}:")
                    print(f"Allocated memory: {allocated_memory:.2f} GB")
                    print(f"Reserved memory: {reserved_memory:.2f} GB")
                    print(
                        f"Free memory: {torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)} bytes")
                    torch.cuda.empty_cache()

                    # set_trace()
                    # backward pass to compute gradients for saliency maps
                    outputs.backward()  # if outputs is not scalar, reduce it to scalar

                    # saliencies = []  # list of saliency maps corresponding to each image in `x_wsi`
                    patch_idx = 0
                    max_patches = 10
                    print("OBTAINING SALIENCY MAPS")
                    for image in x_wsi:
                        print(f"Generating saliency map for patch index {patch_idx} out of {max_patches}")
                        if patch_idx >= max_patches:  # limit to 10 patches
                            break
                        if image.grad is not None:
                            saliency, _ = torch.max(image.grad.data.abs(), dim=1)
                            # saliencies.append(saliency)
                            plot_saliency_maps(saliency, image, tcga_id, patch_idx, output_dir)
                            del saliency
                        else:
                            raise RuntimeError("Gradients have not been computed for one of the images in x_wsi.")
                        patch_idx += 1
                    # plot_saliency_maps(saliencies, x_wsi, tcga_id)

            if opt.calc_IG is True:
                integrated_grads = calc_integrated_gradients(model, opt, tcga_id, x_omic, x_wsi, baseline=baseline,
                                                             steps=10)
                save_path = os.path.join(output_dir_IG, f"integrated_grads_{tcga_id[0]}.npy")
                np.save(save_path, integrated_grads)
                print(f"Saved integrated gradients for {tcga_id[0]} to {save_path}")
            # set_trace()

            # loss = cox_loss(outputs.squeeze(),
            #                 # predictions are not survival outcomes, rather log-risk scores beta*X
            #                 days_to_event,
            #                 event_occurred)  # Cox partial likelihood loss for survival outcome prediction

            # print("\n loss (test): ", loss.data.item())
            # test_loss_epoch += loss.data.item() * len(tcga_id)
            all_predictions.append(outputs.squeeze().detach().cpu().numpy())
            # all_predictions.append(outputs.squeeze())
            del outputs
            torch.cuda.empty_cache()
            all_tcga_ids.append(tcga_id)
            all_times.append(days_to_event)
            all_events.append(event_occurred)
            model.zero_grad()
            torch.cuda.empty_cache()
        # set_trace()
        # # test_loss = test_loss_epoch / len(test_loader.dataset)
        # set_trace()
        # all_predictions = torch.stack(all_predictions)
        # all_times = torch.stack(all_times)
        # all_events = torch.stack(all_events)
        #
        # # convert to numpy arrays for CI calculation
        # all_predictions_np = all_predictions.cpu().numpy()
        # all_times_np = all_times.cpu().numpy()
        # all_events_np = all_events.cpu().numpy()
        # # set_trace()
        # # c_index = concordance_index_censored(all_events_np.astype(bool), all_times_np, all_predictions_np)
        # c_index = concordance_index_censored(all_events_np.astype(bool).flatten(),
        #                                      all_times_np.flatten(),
        #                                      all_predictions_np)
        # # # print(f"Test loss: {test_loss}, CI: {c_index[0]}")

        all_predictions_np = [pred.item() for pred in all_predictions]
        all_events_np = torch.stack(all_events).cpu().numpy()
        all_events_bool_np = all_events_np.astype(bool)
        all_times_np = torch.stack(all_times).cpu().numpy()

        c_index = concordance_index_censored(all_events_bool_np.ravel(), all_times_np.ravel(), all_predictions_np)

        print(f"CI: {c_index[0]}")

    set_trace()
    # stratify based on the median risk scores
    median_prediction = np.median(all_predictions_np)
    high_risk_idx = all_predictions_np >= median_prediction
    low_risk_idx = all_predictions_np < median_prediction

    # separate the times and events into high and low-risk groups
    high_risk_times = all_times_np[high_risk_idx]
    high_risk_events = all_events_np[high_risk_idx]
    low_risk_times = all_times_np[low_risk_idx]
    low_risk_events = all_events_np[low_risk_idx]

    # initialize the Kaplan-Meier fitter
    kmf_high_risk = KaplanMeierFitter()
    kmf_low_risk = KaplanMeierFitter()

    # fit
    kmf_high_risk.fit(high_risk_times, event_observed=high_risk_events, label='High Risk')
    kmf_low_risk.fit(low_risk_times, event_observed=low_risk_events, label='Low Risk')

    # perform the log-rank test
    log_rank_results = logrank_test(high_risk_times, low_risk_times,
                                    event_observed_A=high_risk_events,
                                    event_observed_B=low_risk_events)

    p_value = log_rank_results.p_value
    print(f"Log-Rank Test p-value: {p_value}")
    print(f"Log-Rank Test statistic: {log_rank_results.test_statistic}")

    plt.figure(figsize=(10, 6))
    kmf_high_risk.plot(ci_show=True, color='blue')
    kmf_low_risk.plot(ci_show=True, color='red')
    plt.title(
        'Patient stratification: high risk vs low risk groups based on predicted risk scores\nLog-rank test p-value: {:.4f}'.format(
            p_value))
    plt.xlabel('Time (days)')
    plt.ylabel('Survival probability')
    plt.legend()
    plt.savefig('km_plot_joint_fusion.png', format='png', dpi=300)
    # plt.show()

    # set_trace()

    # # the flow of the gradients in backprop should be through the downstream MLP and the omic MLP
    #
    # # saliency = Saliency(model.wsi_net.forward)
    # # integrated_gradients = IntegratedGradients(model.omic_net.forward)
    # saliency = Saliency(model.forward)
    # integrated_gradients = IntegratedGradients(model.forward)
    #
    # all_attributions_saliency = []
    # all_attributions_ig = []
    #
    # with torch.no_grad():
    #     for tcga_id, days_to_event, event_occurred, x_wsi, x_omic in test_loader:
    #         x_wsi = [x.to(device) for x in x_wsi]
    #         x_omic = x_omic.to(device)
    #         days_to_event = days_to_event.to(device)
    #         event_occurred = event_occurred.to(device)
    #
    #         predictions = model(opt,
    #                             tcga_id,
    #                             x_wsi=x_wsi,
    #                             x_omic=x_omic)
    #
    #         risk_scores = predictions.squeeze()
    #
    #         dataset = CustomDatasetWSI(x_wsi, transform=None)
    #         tile_loader = DataLoader(dataset, batch_size=1,
    #                                  shuffle=False)  # check later why the batch size is hard-coded to 1
    #
    #         if model.module.mode == 'wsi_omic':
    #             # saliency maps for WSI
    #             # generate saliency map for each tile
    #             for tiles in tile_loader:
    #                 tiles = tiles.to(device)
    #                 tiles = tiles.squeeze(0)  # Remove the batch dimension: [8, 3, 256, 256]
    #
    #                 for j, tile in enumerate(tiles):
    #                     tile = tile.unsqueeze(0)  # add batch dimension back
    #                     tile.requires_grad_()
    #                     # set_trace()
    #                     salience_attributions = saliency.attribute(tile, target=risk_scores[j],
    #                                                                additional_forward_args=(
    #                                                                opt, tcga_id, tile, x_omic[j]))
    #
    #                     saliency_attributions_abs = torch.abs(saliency_attributions)
    #                     saliency_map = saliency_attributions_abs.cpu().numpy().squeeze().transpose(1, 2,
    #                                                                                                0)  # HWC format
    #
    #                     # set_trace()
    #             x_wsi_stacked = torch.stack(x_wsi).requires_grad_(True)
    #             x_omic.requires_grad = False  # for WSI, the output gradients should be calculated only wrt the WSI inputs and not the RNASeq inputs
    #
    #             # wsi_embedding = model.wsi_net(x_wsi_stacked)
    #             # omic_embedding = model.omic_net(x_omic)
    #             # combined_embedding = torch.cat((wsi_embedding, omic_embedding), dim=1).requires_grad_(True)
    #             # output = model.fused_mlp(combined_embedding)
    #             # output = output.squeeze()
    #             # output.backward()  # getting gradients of output w.r.t. graph leaves
    #
    #             # saliency_attributions = x_wsi_stacked.grad
    #             saliency_attributions = saliency.attribute(x_wsi_stacked, target=risk_scores,
    #                                                        additional_forward_args=(opt, tcga_id, x_omic))
    #             saliency_attributions_abs = torch.abs(saliency_attributions)
    #             all_attributions_saliency.append(saliency_attributions_abs.cpu().numpy())
    #
    #             # visualize and overlay saliency maps on original patches
    #             for i, attr in enumerate(saliency_attributions_abs):
    #                 attr_np = attr.cpu().numpy().transpose(1, 2, 0)  # convert to HWC format
    #                 attr_np = cv2.normalize(attr_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #                 heatmap = cv2.applyColorMap(attr_np, cv2.COLORMAP_JET)
    #                 original_patch = x_wsi_stacked[i].cpu().numpy().transpose(1, 2, 0)
    #                 original_patch = cv2.normalize(original_patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #
    #                 # overlay the saliency heatmaps with the original WSI patch
    #                 overlay = cv2.addWeighted(heatmap, 0.5, original_patch, 0.5, 0)
    #
    #                 output_path = os.path.join(output_dir, f'saliency_map_patch_{i}.png')
    #                 # cv2.imwrite(output_path, overlay)
    #
    #                 plt.figure(figsize=(6, 6))
    #                 plt.imshow(overlay)
    #                 plt.title(f'Saliency Map Overlay for Patch {i}')
    #                 plt.axis('off')
    #                 plt.savefig(output_path)
    #                 plt.close()
    #
    #         # IG for rnaseq
    #         x_wsi_stacked.requires_grad = False
    #         # x_omic.requires_grad_()
    #         x_omic.requires_grad = True  # reset to True. It was set to False for WSI saliency map computations
    #
    #         baseline = torch.zeros_like(x_omic)  # is zeros the appropriate baseline?
    #         # set_trace()
    #         attrs, delta = integrated_gradients.attribute(inputs=x_omic,
    #                                                       baselines=baseline,
    #                                                       target=risk_scores,
    #                                                       additional_forward_args=(opt, tcga_id, x_omic),
    #                                                       return_convergence_delta=True)
    #         all_attributions_ig.append(attrs.cpu().numpy())
    #
    # all_attributions_saliency = np.concatenate(all_attributions_saliency, axis=0)
    # all_attributions_ig = np.concatenate(all_attributions_ig, axis=0)
    #
    # np.save("wsi_attributions_saliency.npy", all_attributions_saliency)
    # np.save("omic_attributions_ig.npy", all_attributions_ig)
    #
    # return all_attributions_saliency, all_attributions_ig

    return None


if __name__ == "__main__":
    # for inference
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mapping_data_path', type=str,
                        default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/joint_fusion/',
                        help='Path to input mapping data file')
    parser.add_argument('--input_wsi_path', type=str,
                        default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_13july/tiles/256px_9.9x/combined/',
                        help='Path to input WSI tiles')
    # parser.add_argument('--input_wsi_embeddings_path', type=str,
    #                     default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/early_fusion_inputs/',
    #                     help='Path to WSI embeddings generated from pretrained pathology foundation model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size for validation')
    # parser.add_argument('--test_batch_size', type=int, default=1000, help='Batch size for testing (use all samples)')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--input_size_wsi', type=int, default=256, help="input_size for path images")
    parser.add_argument('--embedding_dim_wsi', type=int, default=384, help="embedding dimension for WSI")
    parser.add_argument('--embedding_dim_omic', type=int, default=256, help="embedding dimension for omic")
    parser.add_argument('--input_mode', type=str, default="wsi_omic", help="wsi, omic, wsi_omic")
    parser.add_argument('--fusion_type', type=str, default="joint", help="early, late, joint, joint_omic, unimodal")
    parser.add_argument('--calc_saliency_maps', type=bool, default=True,
                        help="whether to calculate saliency maps for WSI patches")
    parser.add_argument('--calc_IG', type=bool, default=True, help="whether to calculate IG for RNASeq data")
    # Note: True/False have some issues when used in command line
    opt = parser.parse_args()

    # mapping_df = pd.read_json(opt.input_mapping_data_path + "mapping_df.json", orient='index')

    # get predictions on test data, and calculate interpretability metrics
    model = MultimodalNetwork(embedding_dim_wsi=opt.embedding_dim_wsi,
                              embedding_dim_omic=opt.embedding_dim_omic,
                              mode=opt.input_mode,
                              fusion_type=opt.fusion_type)

    model = torch.nn.DataParallel(model)

    # model should return None for the absent modality in the unimodal case

    model.to(device)
    cox_loss = CoxLoss()
    # model.load_state_dict(torch.load("./saved_models/model_epoch_98.pt"))
    # model.load_state_dict(torch.load("./saved_models/model_epoch_8.pt"))
    # model.load_state_dict(torch.load("./saved_models_4sep/model_epoch_20.pt"))

    # model.load_state_dict(torch.load("./saved_models_24_09_05_06_11/model_epoch_21.pt"))
    # model.load_state_dict(torch.load("./saved_models_24_09_05_18_21/model_epoch_44.pt"))  # start LR = 1e-3
    model.load_state_dict(torch.load("./saved_models_24_09_05_23_49/model_epoch_169.pt"))  # 169 # start LR = 1e-4

    # state_dict = torch.load("./saved_models/model_epoch_98.pt")
    # from collections import OrderedDict
    #
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] if k.startswith('module.') else k  # remove `module.` if present
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)

    train_loader, validation_loader, test_loader = create_data_loaders(opt, 'mapping_data.h5')
    validation_dataset = validation_loader.dataset
    test_dataset = test_loader.dataset

    # create a combined loader (validation + test) as the validation data hasn't been used for HPO during training
    combined_dataset = ConcatDataset([validation_dataset, test_dataset])
    combined_loader = torch.utils.data.DataLoader(
        dataset=combined_dataset,
        batch_size=opt.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )


    # # get the baseline for x_omic for IG from the training data (mean expression level for all genes)
    # total_x_omic = None
    # total_samples = 0
    #
    # # loop over train_loader and accumulate the omic data
    # for batch_idx, (tcga_id, days_to_event, event_occurred, x_wsi, x_omic) in enumerate(train_loader):
    #     print(f"Looping over batch {batch_idx}")
    #     if total_x_omic is None:
    #         total_x_omic = torch.zeros_like(x_omic)
    #
    #     total_x_omic += x_omic.sum(dim=0)
    #     total_samples += x_omic.size(0)
    #
    # # compute the mean expression for each gene across all samples
    # mean_x_omic = total_x_omic / total_samples

    # get the mean expression level for all genes across all the training samples to be used as baseline for x_omic for IG
    def compute_mean_omic_from_h5(file_name):
        with h5py.File(file_name, 'r') as hdf:
            train_group = hdf['train']
            total_rnaseq_data = None
            total_samples = 0

            for patient_id in train_group.keys():
                patient_group = train_group[patient_id]
                rnaseq_data = patient_group['rnaseq_data'][:]
                if total_rnaseq_data is None:
                    total_rnaseq_data = np.zeros_like(rnaseq_data)
                total_rnaseq_data += rnaseq_data
                total_samples += 1
            mean_rnaseq_data = total_rnaseq_data / total_samples

        return mean_rnaseq_data


    mean_x_omic = compute_mean_omic_from_h5('mapping_data.h5')

    test_and_interpret(opt, model, combined_loader, device, baseline=mean_x_omic)
    # test_and_interpret(opt, model, combined_loader, device, baseline=None)
    # test_and_interpret(opt, model, test_loader, device) # remove cox_loss from the arguments
