import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from sksurv.metrics import concordance_index_censored
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def denormalize_image(image, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    image = (image * std) + mean
    image = np.clip(image, 0, 1)
    return image


def calc_integrated_gradients(
    model, config, tcga_id, x_omic, x_wsi, baseline=None, steps=10
):
    # baseline = None
    # baseline.shape
    # torch.Size([1, 19962])
    # set_trace()
    if baseline is None:
        print("** using trivial zero baseline for IG **")
        baseline = torch.zeros_like(x_omic).to(x_omic.device)
    else:
        print(
            "** Using mean gene expression values over training samples as the baseline for IG **"
        )
        baseline = torch.from_numpy(baseline).to(x_omic.device)
        baseline = baseline.float()
        baseline = baseline * torch.ones_like(x_omic).to(x_omic.device)
    # set_trace()
    print(f"CALCULATING INTEGRATED GRADIENTS OVER {steps} steps")
    scaled_inputs = [
        baseline + (float(i) / steps) * (x_omic - baseline) for i in range(steps + 1)
    ]
    gradients = []
    steps_index = 0
    for scaled_input in scaled_inputs:
        print("steps_index: ", steps_index)
        scaled_input = scaled_input.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            pred, _, _ = model(
                config,
                tcga_id,
                x_wsi=x_wsi,  # list of tensors (one for each tile)
                x_omic=scaled_input,
            )
            print("prediction: ", pred)
            output = pred.sum()
            output.backward()
        gradients.append(scaled_input.grad.detach().cpu().numpy())
        steps_index += 1
    # set_trace()
    avg_gradients = np.mean(gradients[:-1], axis=0)
    integrated_grads = (
        x_omic.detach().cpu().numpy() - baseline.detach().cpu().numpy()
    ) * avg_gradients

    return integrated_grads


def plot_saliency_maps(
    saliencies,
    x_wsi,
    tcga_id,
    patch_id,
    output_dir,
    threshold=0.8,
):
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
    # saliency_overlay = ax.imshow(saliency, cmap="hot", alpha=0.5)
    saliency_overlay = ax.imshow(np.clip(saliency, threshold, 1), cmap="hot", alpha=0.5)
    cbar = plt.colorbar(saliency_overlay, ax=ax)
    cbar.set_label("saliency value", rotation=270, labelpad=15)

    plt.title("saliency map overlay")

    # plot the saliency map alone
    plt.subplot(1, 3, 3)
    plt.imshow(saliency, cmap="hot")
    plt.colorbar(label="saliency value")
    plt.title("saliency map only")

    save_path = os.path.join(
        output_dir, f"saliency_overlay_{tcga_id[0]}_{patch_id}.png"
    )
    plt.savefig(save_path)
    print(f"saved saliency overlay to {save_path}")

    plt.close()


def interpret_omic():

    # not used since the code is already simple enough

    pass


def interpret_wsi(x_wsi, tcga_id, output_dir_saliency):

    patch_idx = 0
    max_patches = 10
    print("OBTAINING SALIENCY MAPS")
    for image in x_wsi:
        print(
            f"Generating saliency map for patch index {patch_idx} out of {max_patches}"
        )
        if patch_idx >= max_patches:  # limit to 10 patches
            break
        if image.grad is not None:
            saliency, _ = torch.max(image.grad.data.abs(), dim=1)
            # saliencies.append(saliency)
            plot_saliency_maps(saliency, image, tcga_id, patch_idx, output_dir_saliency)
            del saliency
        else:
            raise RuntimeError(
                "Gradients have not been computed for one of the images in x_wsi."
            )
        patch_idx += 1


def test_and_interpret(config, model, test_loader, device, baseline=None):
    model.eval()
    test_loss_epoch = 0.0
    all_tcga_ids = []
    all_predictions = []
    all_times = []
    all_events = []

    # base directory
    import os

    base_path = Path(config.testing.output_base_dir)
    # create the directory to save saliency maps if it doesn't exist
    output_dir_saliency = str(base_path / "saliency_maps_6sep")
    os.makedirs(output_dir_saliency, exist_ok=True)

    output_dir_IG = str(base_path / "IG_6sep")
    os.makedirs(output_dir_IG, exist_ok=True)

    # for training, only the last transformer block (block 11) in the WSI encoder was kept trainable
    # see WSIEncoder class in generate_Wsi_embeddings.py

    # Get the CI and the KM plots for the test set
    excluded_ids = [
        "TCGA-05-4395",
        "TCGA-86-8281",
    ]  # contains anomalous time to event and censoring data
    # remove these ids during the input json/h5 file creation
    with torch.no_grad():
        for batch_idx, (
            tcga_id,
            days_to_event,
            event_occurred,
            x_wsi,
            x_omic,
        ) in enumerate(test_loader):
            if tcga_id[0] in excluded_ids:
                print(f"Skipping TCGA ID: {tcga_id}")
                continue

            x_wsi = [x.to(device) for x in x_wsi]
            x_omic = x_omic.to(device)
            days_to_event = days_to_event.to(device)
            event_occurred = event_occurred.to(device)

            # enable gradients only after data loading
            x_wsi = [x.requires_grad_() for x in x_wsi]

            # print(f"Batch size: {len(test_loader.dataset)}")
            # print(
            #     f"Test Batch index: {batch_idx + 1} out of {np.ceil(len(test_loader.dataset) / opt.test_batch_size)}"
            # )
            # # print("TCGA ID: ", tcga_id)
            # print("Days to event: ", days_to_event)
            # print("event occurred: ", event_occurred)

            if config.testing.calc_saliency_maps is False:
                outputs, wsi_embedding, omic_embedding = model(
                    config,
                    tcga_id,
                    x_wsi=x_wsi,  # list of tensors (one for each tile)
                    x_omic=x_omic,
                )

            else:
                # perform the forward pass without torch.no_grad() to allow gradient computation
                with torch.enable_grad():
                    outputs, wsi_embedding, omic_embedding = model(
                        config,
                        tcga_id,
                        x_wsi=x_wsi,  # list of tensors (one for each tile)
                        x_omic=x_omic,
                    )

                    # Check and print memory usage after each batch
                    allocated_memory = torch.cuda.memory_allocated(device) / (
                        1024**3
                    )  # in GB
                    reserved_memory = torch.cuda.memory_reserved(device) / (
                        1024**3
                    )  # in GB

                    print(f"After batch {batch_idx + 1}:")
                    print(f"Allocated memory: {allocated_memory:.2f} GB")
                    print(f"Reserved memory: {reserved_memory:.2f} GB")
                    print(
                        f"Free memory: {torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)} bytes"
                    )
                    torch.cuda.empty_cache()

                    # set_trace()
                    # backward pass to compute gradients for saliency maps
                    outputs.backward()  # if outputs is not scalar, reduce it to scalar

                    # saliencies = []  # list of saliency maps corresponding to each image in `x_wsi`

                    interpret_wsi(x_wsi, tcga_id, output_dir_saliency)

            if config.testing.calc_IG is True:
                integrated_grads = calc_integrated_gradients(
                    model, config, tcga_id, x_omic, x_wsi, baseline=baseline, steps=10
                )
                save_path = os.path.join(
                    output_dir_IG, f"integrated_grads_{tcga_id[0]}.npy"
                )
                np.save(save_path, integrated_grads)
                print(f"Saved integrated gradients for {tcga_id[0]} to {save_path}")

            all_predictions.append(outputs.squeeze().detach().cpu().numpy())

            del outputs
            torch.cuda.empty_cache()
            all_tcga_ids.append(tcga_id)
            all_times.append(days_to_event)
            all_events.append(event_occurred)
            model.zero_grad()
            torch.cuda.empty_cache()

        all_predictions_np = [pred.item() for pred in all_predictions]
        all_events_np = torch.stack(all_events).cpu().numpy()
        all_events_bool_np = all_events_np.astype(bool)
        all_times_np = torch.stack(all_times).cpu().numpy()

        c_index = concordance_index_censored(
            all_events_bool_np.ravel(), all_times_np.ravel(), all_predictions_np
        )

        print(f"CI: {c_index[0]}")

    # set_trace()
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
    kmf_high_risk.fit(
        high_risk_times, event_observed=high_risk_events, label="High Risk"
    )
    kmf_low_risk.fit(low_risk_times, event_observed=low_risk_events, label="Low Risk")

    # perform the log-rank test
    log_rank_results = logrank_test(
        high_risk_times,
        low_risk_times,
        event_observed_A=high_risk_events,
        event_observed_B=low_risk_events,
    )

    p_value = log_rank_results.p_value
    print(f"Log-Rank Test p-value: {p_value}")
    print(f"Log-Rank Test statistic: {log_rank_results.test_statistic}")

    plt.figure(figsize=(10, 6))
    kmf_high_risk.plot(ci_show=True, color="blue")
    kmf_low_risk.plot(ci_show=True, color="red")
    plt.title(
        "Patient stratification: high risk vs low risk groups based on predicted risk scores\nLog-rank test p-value: {:.4f}".format(
            p_value
        )
    )
    plt.xlabel("Time (days)")
    plt.ylabel("Survival probability")
    plt.legend()
    output_path = str(Path(config.testing.output_base_dir) / "km_plot_joint_fusion.png")
    plt.savefig(output_path, format="png", dpi=300)

    return None
