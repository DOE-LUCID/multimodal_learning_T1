from joint_fusion.models.loss_functions import JointLoss
from joint_fusion.models.multimodal_network import MultimodalNetwork
from joint_fusion.utils.utils import mixed_collate, seed_worker, print_gpu_memory_usage
from joint_fusion.testing.analysis import evaluate_test_set
from .learning_scheduler import CosineAnnealingWarmRestartsDecay
from .pretraining import (
    create_data_loaders,
    create_stratified_survival_folds,
    extract_survival_data,
)
from .visualizations import plot_survival_distributions

import gc
import numpy as np
import torch
import os
import wandb
import time
import joblib
import dataclasses
import logging
from datetime import datetime

from torch import nn
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler

from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.autograd.set_detect_anomaly(True)

current_time = datetime.now().strftime("%y_%m_%d_%H_%M")


def train_observ_test(config, h5_file, device):

    from pathlib import Path
    from dotenv import load_dotenv

    env_path: Path = Path.cwd() / ".env"

    load_dotenv(env_path)

    project = os.getenv("WANDB_PROJECT")
    entity = os.getenv("WANDB_ENTITY")

    wandb.init(
        project=project,
        entity=entity,
        config=dataclasses.asdict(config),
    )

    current_time = datetime.now()

    train_loader, validation_loader, test_loader = create_data_loaders(config, h5_file)

    validation_dataset = validation_loader.dataset
    test_dataset = test_loader.dataset

    # create a combined loader (validation + test) as the validation data hasn't been used for HPO during training
    test_dataset = ConcatDataset([validation_dataset, test_dataset])
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.testing.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    dataset = (
        train_loader.dataset
    )  # use only the training dataset for CV during training [both the validation and test splits can be used for testing]

    total_samples = len(dataset)
    logging.info(f"total training data size: {total_samples} samples")

    # Extract survival data for stratification
    logging.info("Extracting survival data for startified CV...")
    times, events = extract_survival_data(dataset)

    print(f"Survival data summary:")
    print(f"  - Total samples: {len(times)}")
    print(f"  - Events: {events.sum()} ({events.mean():.1%})")
    print(f"  - Censored: {len(events) - events.sum()} ({1 - events.mean():.1%})")
    print(f"  - Median survival time: {np.median(times):.1f} days")
    print(f"  - Time range: {times.min():.1f} - {times.max():.1f} days")

    try:
        folds = create_stratified_survival_folds(
            times=times,
            events=events,
            n_splits=config.training.n_folds,
            n_time_bins=10,
        )
        logging.info(f"Successfully created {len(folds)} stratified folds")

    except Exception as e:
        print(f"Error creating stratified folds: {e}")
        # Fallback to regular k-fold if stratified fails
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=config.training.n_folds, shuffle=True, random_state=40)
        folds = list(kf.split(np.arange(len(dataset))))
        print("Falling back to regular K-Fold CV")

    # need to keep fold index at 0 for the sake of the averaging by epoch
    if config.training.use_mixed_precision:
        amp_scaler = GradScaler()
        print("Using mixed precision training with StandardScaler")

    # Step counter for wandb
    step_counter = 0
    # Main fold iteration loop
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{len(folds)}")
        print(f"{'='*60}")

        # Print fold statistics
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")

        # Extract survival data for this fold to verify stratification
        times_train_fold = times[train_idx]
        events_train_fold = events[train_idx]
        times_val_fold = times[val_idx]
        events_val_fold = events[val_idx]

        train_event_rate = events_train_fold.mean()
        val_event_rate = events_val_fold.mean()

        print(f"Train event rate: {train_event_rate:.3f}")
        print(f"Validation event rate: {val_event_rate:.3f}")
        print(f"Train median time: {np.median(times_train_fold):.1f} days")
        print(f"Validation median time: {np.median(times_val_fold):.1f} days")

        print_gpu_memory_usage()  # Monitor memory at start of each fold

        if config.training.plot_survival_distributions:
            plot_survival_distributions(
                times_train_fold,
                events_train_fold,
                times_val_fold,
                events_val_fold,
                fold_idx,
                save_dir=os.path.join(
                    config.logging.checkpoint_dir, "fold_distributions"
                ),
            )

        # Create subsets for this fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create data loaders for this fold
        train_loader_fold = DataLoader(
            train_subset,
            batch_size=config.training.batch_size,
            collate_fn=mixed_collate,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            worker_init_fn=seed_worker,
        )
        val_loader_fold = DataLoader(
            val_subset,
            collate_fn=mixed_collate,
            batch_size=config.training.val_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

        # Compute number of batches per epoch
        num_batches_per_epoch = len(train_loader_fold)
        print(
            f"Number of batches per epoch for fold {fold_idx + 1}: {num_batches_per_epoch}"
        )

        # Update wandb config
        wandb.config.update(
            {f"num_batches_per_epoch_fold_{fold_idx}": num_batches_per_epoch},
            allow_val_change=True,
        )

        model = MultimodalNetwork(
            embedding_dim_wsi=config.model.embedding_dim_wsi,
            embedding_dim_omic=config.model.embedding_dim_omic,
            mode=config.model.input_mode,
            fusion_type=config.model.fusion_type,
            joint_embedding_type=config.model.joint_embedding,
            mlp_layers=config.model.mlp_layers,
            dropout=config.model.dropout,
        )

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
            torch.cuda.set_device(0)

        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.training.lr, weight_decay=1e-4
        )

        if config.scheduler.type == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, **config.scheduler.exponential
            )
        elif config.scheduler.type == "cosine_warmer":
            scheduler = CosineAnnealingWarmRestartsDecay(
                optimizer, **config.scheduler.cosine_warmer
            )
        elif config.scheduler.type == "step_lr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, **config.scheduler.step_lr
            )
        else:
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )

        joint_loss = JointLoss(
            sim_weight=config.training.sim_loss_weight,
            contrast_weight=config.training.contrast_loss_weight,
            contrast_wsi_weight=config.training.contrast_wsi_weight,
            contrast_omic_weight=config.training.contrast_omic_weight,
            contrast_joint_weight=config.training.contrast_joint_weight,
        )

        # Initialize and fit the scaler for this fold
        print(f"Fitting scaler for fold {fold_idx + 1}...")
        scaler = StandardScaler()
        for batch_idx, (tcga_id, _, _, _, x_omic) in enumerate(train_loader_fold):
            if batch_idx % 10 == 0:  # Print progress every 10 batches
                print(f"  Fitting batch {batch_idx + 1}/{len(train_loader_fold)}")

            x_omic_np = x_omic.cpu().numpy()
            nan_mask = np.isnan(x_omic_np)
            inf_mask = np.isinf(x_omic_np)
            logging.info(
                "NaN count: %d | Inf count: %d", nan_mask.sum(), inf_mask.sum()
            )

            logging.info(
                "%d zero-variance columns in this batch",
                (x_omic_np.std(axis=0) == 0).sum(),
            )

            scaler.partial_fit(x_omic_np)

        # Save the scaler for the current fold
        scaler_path = os.path.join(
            config.logging.checkpoint_dir, f"scaler_fold_{fold_idx}.save"
        )
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved for fold {fold_idx + 1} at {scaler_path}")

        # EPOCH TRAINING LOOP for this fold
        for epoch in range(config.training.num_epochs):
            print(
                f"\n--- Fold {fold_idx + 1}/{len(folds)}, Epoch {epoch + 1}/{config.training.num_epochs} ---"
            )
            start_train_time = time.time()
            model.train()

            loss_epoch = 0

            # Add debug info for first batch
            print(f"Starting epoch {epoch}, about to iterate over training batches...")
            print(f"Training loader has {len(train_loader_fold)} batches")

            # model training in batches for the train dataloader for the current fold
            for batch_idx, (
                tcga_id,
                days_to_event,
                event_occurred,
                x_wsi,
                x_omic,
            ) in enumerate(train_loader_fold):
                # x_wsi is a list of tensors (one tensor for each tile)
                print(
                    f"Total training samples in fold: {len(train_loader_fold.dataset)}"
                )
                print(f"Batch size: {config.training.batch_size}")
                print(f"Batch index: {batch_idx} out of {num_batches_per_epoch}")

                # NOTE: Do not apply standard scaler to omic data
                # x_wsi = [
                #     [tile.to(device) for tile in patient_tiles]
                #     for patient_tiles in x_wsi
                # ]
                x_wsi = [x.to(device) for x in x_wsi]
                x_omic = x_omic.to(device)

                print(f"After scaling - x_omic shape: {x_omic.shape}")
                days_to_event = days_to_event.to(device)
                event_occurred = event_occurred.to(device)

                print("Days to event: ", days_to_event)
                print("event occurred: ", event_occurred)

                optimizer.zero_grad()

                if config.training.use_mixed_precision:
                    with autocast():  # should wrap only the forward pass including the loss calculation
                        (
                            predictions,
                            wsi_embedding,
                            omic_embedding,
                            combined_embedding,
                        ) = model(
                            config,
                            tcga_id,
                            x_wsi=x_wsi,
                            x_omic=x_omic,  # Now properly scaled
                        )

                        predictions = predictions.view(-1)

                        # print(f"Model predictions shape: {predictions.shape}")
                        loss = joint_loss(
                            predictions,
                            days_to_event,
                            event_occurred,
                            wsi_embedding,
                            omic_embedding,
                            combined_embedding,
                        )
                        print("\n loss (train, mixed precision): ", loss.data.item())
                        loss_epoch += loss.data.item()

                    # Mixed precision backward pass
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(optimizer)
                    amp_scaler.update()

                    del predictions, wsi_embedding, omic_embedding, loss, x_wsi, x_omic

                    if batch_idx % 5 == 0:
                        torch.cuda.empty_cache()

                else:
                    print(" Not using mixed precision")
                    # model for survival outcome (uses Cox PH partial log likelihood as the loss function)
                    # the model output should be considered as beta*X to be used in the Cox loss function

                    start_time = time.time()

                    predictions, wsi_embedding, omic_embedding, combined_embedding = (
                        model(
                            config,
                            tcga_id,
                            x_wsi=x_wsi,  # list of tensors (one for each tile)
                            x_omic=x_omic,
                        )
                    )

                    predictions = predictions.view(-1)
                    # print(f"predictions: {predictions} from train_test.py")
                    step1_time = time.time()
                    loss = joint_loss(
                        predictions,
                        # predictions are not survival outcomes, rather log-risk scores beta*X
                        days_to_event,
                        event_occurred,
                        wsi_embedding,
                        omic_embedding,
                        combined_embedding,
                    )  # Cox partial likelihood loss for survival outcome prediction
                    print("\n loss (train): ", loss.data.item())
                    step2_time = time.time()
                    loss_epoch += (
                        loss.data.item()
                    )  # * len(tcga_id)  # multiplying loss by batch size for accurate epoch averaging
                    # backpropagate loss through the entire model arch upto the inputs
                    loss.backward(
                        retain_graph=True if epoch == 0 and batch_idx == 0 else False
                    )  # tensors retained to allow backpropagation for torchhviz (for visualizing the graph)
                    optimizer.step()

                    torch.cuda.empty_cache()

                    step3_time = time.time()
                    print(
                        f"(in train_nn) Step 1: {step1_time - start_time:.4f}s, Step 2: {step2_time - step1_time:.4f}s, Step 3: {step3_time - step2_time:.4f}s"
                    )

            train_loss = loss_epoch / len(
                train_loader_fold.dataset
            )  # average training loss per sample for the epoch

            scheduler.step()  # step scheduler after each epoch
            print("\n train loss over epoch: ", train_loss)
            end_train_time = time.time()
            train_duration = end_train_time - start_train_time

            # return here for profile
            if config.logging.profile:
                return model, optimizer
            # check validation for all epochs >= 0
            if epoch >= 0:

                # save model once every 5 epochs
                if (epoch + 1) % 5 == 0 and epoch > 0:
                    checkpoint_path = os.path.join(
                        config.logging.checkpoint_dir,
                        f"checkpoint_fold_{fold_idx}_epoch_{epoch}.pth",
                    )
                    torch.save(
                        {
                            "epoch": epoch,
                            "fold": fold_idx,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        checkpoint_path,
                    )
                    logging.info(
                        f"Checkpoint saved at epoch {epoch}, fold {fold_idx}, to {checkpoint_path}"
                    )

                # before validation to get the dynamic weights, but only if the mode is "wsi_omic"

                start_val_time = time.time()

                # get predictions on the validation dataset
                model.eval()
                val_loss_epoch = 0.0
                val_loss = 0.0
                val_ci = 0.0
                val_event_rate = 0.0
                val_duration = 0.0

                val_predictions = []
                val_times = []
                val_events = []
                val_wsi_emb = []
                val_omic_emb = []
                val_tcga_id_list = []

                test_ci = 0.0

                with torch.no_grad():  # inference on the validation data
                    for batch_idx, (
                        tcga_id,
                        days_to_event,
                        event_occurred,
                        x_wsi,
                        x_omic,
                    ) in enumerate(val_loader_fold):
                        # x_wsi is a list of tensors (one tensor for each tile)
                        print(f"Batch size: {len(val_loader_fold.dataset)}")
                        print(
                            f"Validation Batch index: {batch_idx + 1} out of {np.ceil(len(val_loader_fold.dataset) / config.training.val_batch_size)}"
                        )
                        x_wsi = [x.to(device) for x in x_wsi]
                        x_omic = x_omic.to(device)

                        days_to_event = days_to_event.to(device)
                        event_occurred = event_occurred.to(device)
                        print("Days to event: ", days_to_event)
                        print("event occurred: ", event_occurred)
                        batch_size = len(tcga_id)

                        if batch_size < torch.cuda.device_count() and isinstance(
                            model, nn.DataParallel
                        ):
                            (
                                outputs,
                                wsi_embedding,
                                omic_embedding,
                                combined_embedding,
                            ) = model.module(
                                config, tcga_id, x_wsi=x_wsi, x_omic=x_omic
                            )
                        else:
                            (
                                outputs,
                                wsi_embedding,
                                omic_embedding,
                                combined_embedding,
                            ) = model(
                                config,
                                tcga_id,
                                x_wsi=x_wsi,  # list of tensors (one for each tile)
                                x_omic=x_omic,
                            )

                        outputs = outputs.view(-1)
                        loss = joint_loss(
                            outputs,
                            # predictions are not survival outcomes, rather log-risk scores beta*X
                            days_to_event,
                            event_occurred,
                            wsi_embedding,
                            omic_embedding,
                            combined_embedding,
                        )  # Cox partial likelihood loss for survival outcome prediction
                        logging.info(
                            f"\n loss (validation): {loss.data.item()}",
                        )
                        val_loss_epoch += loss.data.item() * len(tcga_id)
                        val_predictions.append(outputs)
                        val_times.append(days_to_event)
                        val_events.append(event_occurred)

                        val_wsi_emb.append(wsi_embedding)
                        val_omic_emb.append(omic_embedding)
                        val_tcga_id_list.extend(tcga_id)

                    torch.cuda.empty_cache()

                    val_loss = val_loss_epoch / len(val_loader_fold.dataset)

                    val_predictions = torch.cat(val_predictions)
                    val_times = torch.cat(val_times)
                    val_events = torch.cat(val_events)

                    wsi_embeddings = torch.cat(val_wsi_emb, dim=0)
                    omic_embeddings = torch.cat(val_omic_emb, dim=0)

                    # convert to numpy arrays for CI calculation
                    val_predictions_np = val_predictions.cpu().numpy()
                    val_times_np = val_times.cpu().numpy()
                    val_events_np = val_events.cpu().numpy()
                    val_event_rate = val_events_np.mean()

                    wsi_np = wsi_embeddings.cpu().numpy()
                    omic_np = omic_embeddings.cpu().numpy()
                    tcga_id_np = np.array(val_tcga_id_list)

                    val_ci = concordance_index_censored(
                        val_events_np.astype(bool), val_times_np, val_predictions_np
                    )[0]
                    print(f"Validation loss: {val_loss}, CI: {val_ci}")

                    end_val_time = time.time()
                    val_duration = end_val_time - start_val_time

                    # save embeddings once every 5 epochs
                    if (epoch + 1) % 5 == 0 and epoch > 0:
                        wsi_checkpoint_path = os.path.join(
                            config.logging.checkpoint_dir,
                            f"wsi_embedding_fold_{fold_idx}_epoch_{epoch}.npy",
                        )
                        np.save(wsi_checkpoint_path, wsi_np)

                        omic_checkpoint_path = os.path.join(
                            config.logging.checkpoint_dir,
                            f"omic_embedding_fold_{fold_idx}_epoch_{epoch}.npy",
                        )

                        np.save(omic_checkpoint_path, omic_np)

                        print(
                            f"Embeddings saved at epoch {epoch}, fold {fold_idx}, to {wsi_checkpoint_path} and {omic_checkpoint_path}"
                        )

                        id_checkpoint_path = os.path.join(
                            config.logging.checkpoint_dir,
                            f"tcga_ids_fold_{fold_idx}_epoch_{epoch}.npy",
                        )

                        np.save(id_checkpoint_path, tcga_id_np)

                test_ci, test_event_rate, test_statistic, p_value = evaluate_test_set(
                    model, test_loader, device, config
                )

                torch.cuda.empty_cache()
                gc.collect()

                mode = (
                    model.module.mode
                    if isinstance(model, nn.DataParallel)
                    else model.mode
                )
                actual_model = model.module if hasattr(model, "module") else model
                if mode == "wsi_omic":
                    wsi_weight_val = (
                        actual_model.wsi_weight.item()
                        if isinstance(actual_model.wsi_weight, torch.Tensor)
                        else actual_model.wsi_weight
                    )
                    omic_weight_val = (
                        actual_model.omic_weight.item()
                        if isinstance(actual_model.omic_weight, torch.Tensor)
                        else actual_model.omic_weight
                    )
                    epoch_metrics = {
                        # Losses
                        "Loss/train_epoch": train_loss,
                        "Loss/val_epoch": val_loss,
                        # Performance metrics
                        "CI/validation": val_ci,
                        "CI/test": test_ci,
                        "pvalue/test": p_value,
                        "test_statistic/test": test_statistic,
                        "Event_rate/validation": val_event_rate,
                        "Event_rate/test": test_event_rate,
                        # Time tracking
                        "Time/train_epoch": train_duration,
                        "Time/val_epoch": val_duration,
                        "Time/total_epoch": train_duration + val_duration,
                        # Learning rate
                        "LR": optimizer.param_groups[0]["lr"],
                        # Metadata
                        "fold": fold_idx,
                        "epoch": epoch,
                        "wsi_weight": wsi_weight_val,
                        "omic_weight": omic_weight_val,
                    }
                else:
                    # is this really necessary if only focus on joint fusion model? Then the mode can just be removed
                    epoch_metrics = {
                        # Losses
                        "Loss/train_epoch": train_loss,
                        "Loss/val_epoch": val_loss,
                        # Performance metrics
                        "CI/validation": val_ci,
                        "CI/test": test_ci,
                        "Event_rate/validation": val_event_rate,
                        "Event_rate/test": test_event_rate,
                        # Time tracking
                        "Time/train_epoch": train_duration,
                        "Time/val_epoch": val_duration,
                        "Time/total_epoch": train_duration + val_duration,
                        # Learning rate
                        "LR": optimizer.param_groups[0]["lr"],
                        # Metadata
                        "fold": fold_idx,
                        "epoch": epoch,
                    }
                # Log all metrics
                wandb.log(epoch_metrics, step=step_counter)

                # increment step counter
                step_counter += 1

        torch.cuda.empty_cache()
        gc.collect()
        logging.info(
            f"Memory cleared after fold {fold_idx} - model will be recreated for next fold"
        )

    wandb.finish()
    return model, optimizer
