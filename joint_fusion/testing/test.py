import np
import torch


def interpret_omic():

    pass


def interpret_wsi(x_wsi):

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


def test_and_interpret(opt, model, test_loader, device, baseline=None):
    model.eval()
    test_loss_epoch = 0.0
    all_tcga_ids = []
    all_predictions = []
    all_times = []
    all_events = []

    # base directory
    import os

    base_path = Path(opt.output_base_dir)
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

            if opt.calc_saliency_maps is False:
                outputs, wsi_embedding, omic_embedding = model(
                    opt,
                    tcga_id,
                    x_wsi=x_wsi,  # list of tensors (one for each tile)
                    x_omic=x_omic,
                )

            if opt.calc_saliency_maps is True:
                # perform the forward pass without torch.no_grad() to allow gradient computation
                with torch.enable_grad():
                    outputs, wsi_embedding, omic_embedding = model(
                        opt,
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
                            plot_saliency_maps(
                                saliency, image, tcga_id, patch_idx, output_dir_saliency
                            )
                            del saliency
                        else:
                            raise RuntimeError(
                                "Gradients have not been computed for one of the images in x_wsi."
                            )
                        patch_idx += 1
                    # plot_saliency_maps(saliencies, x_wsi, tcga_id)

            if opt.calc_IG is True:
                integrated_grads = calc_integrated_gradients(
                    model, opt, tcga_id, x_omic, x_wsi, baseline=baseline, steps=10
                )
                save_path = os.path.join(
                    output_dir_IG, f"integrated_grads_{tcga_id[0]}.npy"
                )
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
    output_path = str(Path(opt.output_base_dir) / "km_plot_joint_fusion.png")
    plt.savefig(output_path, format="png", dpi=300)

    return None
