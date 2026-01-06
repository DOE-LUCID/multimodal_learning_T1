import numpy as np
import torch
from torch import nn

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored


def evaluate_test_set(model, test_loader, device, config, excluded_ids=None):
    """
    Shared function for test set evaluation
    """
    if excluded_ids is None:
        excluded_ids = ["TCGA-05-4395", "TCGA-86-8281"]

    # Extract the base model from DataParallel wrapper if it exists
    if isinstance(model, nn.DataParallel):
        test_model = model.module
        print("Removed DataParallel wrapper for testing")
    else:
        test_model = model

    # Move to single GPU and ensure it's in eval mode
    test_model = test_model.to(device)
    test_model.eval()

    test_predictions = []
    test_times = []
    test_events = []

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

            print(f"Batch size: {len(test_loader.dataset)}")
            print(
                f"Test Batch index: {batch_idx + 1} out of {np.ceil(len(test_loader.dataset) / config.testing.test_batch_size)}"
            )
            print("TCGA ID: ", tcga_id)
            print("Days to event: ", days_to_event)
            print("event occurred: ", event_occurred)

            outputs, _, _ = test_model(config, tcga_id, x_wsi=x_wsi, x_omic=x_omic)

            # Collect results consistently
            test_predictions.append(outputs.squeeze().detach().cpu().numpy())
            test_times.append(days_to_event.cpu().numpy())
            test_events.append(event_occurred.cpu().numpy())

    # Process results consistently
    all_predictions_np = [np.asarray(pred).flatten()[0] for pred in test_predictions]
    all_events_np = np.concatenate(test_events)
    all_times_np = np.concatenate(test_times)
    test_event_rate = all_events_np.mean()

    # Safe CI calculation
    try:
        test_ci = concordance_index_censored(
            all_events_np.astype(bool), all_times_np, all_predictions_np
        )[0]
        print(f"Test CI: {test_ci}")
    except Exception as e:
        print(f"Could not calculate test CI: {e}")
        test_ci = float("nan")

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
    return test_ci, test_event_rate, log_rank_results.test_statistic, p_value
