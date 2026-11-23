import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging

from pathlib import Path
from sksurv.metrics import concordance_index_censored
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from joint_fusion.utils.utils import parse_tile_coordinates

logger = logging.getLogger(__name__)


def setup_output_dirs(base_path):
    output_dirs = {
        "saliency": str(base_path / "saliency_maps_6sep"),
        "ig": str(base_path / "IG_6sep"),
        "attention": str(base_path / "attention_scores"),
        "attention_tile_scores": str(base_path / "attention_tile_scores"),
        "saliency_tile_scores": str(base_path / "saliency_tile_scores"),
    }
    for output_dir in output_dirs.values():
        os.makedirs(output_dir, exist_ok=True)
    return output_dirs


def unpack_test_batch(batch):
    if len(batch) == 6:
        tcga_id, days_to_event, event_occurred, x_wsi, x_omic, tile_names = batch
        tile_names = list(tile_names[0])
    else:
        tcga_id, days_to_event, event_occurred, x_wsi, x_omic = batch
        tile_names = None

    return tcga_id, days_to_event, event_occurred, x_wsi, x_omic, tile_names


def move_batch_to_device(x_wsi, x_omic, days_to_event, event_occurred, device):
    return (
        x_wsi.to(device),
        x_omic.to(device),
        days_to_event.to(device),
        event_occurred.to(device),
    )


def validate_attribution_config(config):
    needs_single_item_batches = any(
        (
            config.testing.calc_saliency_maps,
            config.testing.calc_IG,
            getattr(config.testing, "calc_attention_maps", False),
        )
    )
    if needs_single_item_batches and config.testing.test_batch_size != 1:
        raise ValueError(
            "Attribution export currently requires testing.test_batch_size == 1"
        )
    valid_selection = {"all"}
    if config.testing.saliency_tile_selection not in valid_selection:
        raise ValueError(
            f"Unsupported saliency_tile_selection: {config.testing.saliency_tile_selection}"
        )


def denormalize_image(image, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    image = (image * std) + mean
    image = np.clip(image, 0, 1)
    return image


def calc_integrated_gradients(
    model, config, tcga_id, x_omic, x_wsi, baseline=None, steps=50
):
    if baseline is None:
        logger.info("** using trivial zero baseline for IG **")
        baseline = torch.zeros_like(x_omic).to(x_omic.device)
        baseline_kind = "zeros"
    else:
        logger.info(
            "** Using mean gene expression values over training samples as the baseline for IG **"
        )
        baseline = torch.from_numpy(baseline).to(x_omic.device)
        baseline = baseline.float()
        baseline = baseline * torch.ones_like(x_omic).to(x_omic.device)
        baseline_kind = "train_mean_expression"

    logger.info(f"CALCULATING INTEGRATED GRADIENTS OVER {steps} steps")
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        fixed_wsi_embedding = model.encode_wsi(x_wsi).detach()

    # alpha = 0 is the baseline, alpha = 1 is the patient's real expression.
    scaled_inputs = [
        baseline + (float(i) / steps) * (x_omic - baseline) for i in range(steps + 1)
    ]
    gradients = []
    endpoint_preds = {}
    for step_index, scaled_input in enumerate(scaled_inputs):
        scaled_input = scaled_input.clone().detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            pred, _, _, _ = model.forward_with_fixed_wsi(
                fixed_wsi_embedding=fixed_wsi_embedding,
                x_omic=scaled_input,
            )
            output = pred.sum()
            output.backward()
        gradients.append(scaled_input.grad.detach().cpu().numpy())
        if step_index in (0, steps):
            endpoint_preds[step_index] = float(output.detach().cpu())

    # Trapezoidal Riemann sum of the gradient over alpha in [0, 1]. The interval has
    # unit length, so the integral IS the path-averaged gradient. (The previous code
    # took a left-endpoint sum over 10 steps, which biases the integral; trapezoid at
    # the configured step count drives the completeness error below.)
    grad_stack = np.stack(gradients, axis=0)
    path_gradients = (grad_stack[:-1] + grad_stack[1:]).sum(axis=0) / (2.0 * steps)

    x_np = x_omic.detach().cpu().numpy()
    baseline_np = baseline.detach().cpu().numpy()
    integrated_grads = (x_np - baseline_np) * path_gradients

    # Completeness axiom: the attributions must sum to the change in the model output
    # between the baseline and the real input. This is the one sanity check that tells
    # us the attribution is trustworthy, so record it per patient rather than trusting it.
    delta_pred = endpoint_preds[steps] - endpoint_preds[0]
    attribution_sum = float(integrated_grads.sum())
    completeness_error = attribution_sum - delta_pred
    denom = abs(delta_pred) if abs(delta_pred) > 1e-8 else 1.0
    logger.info(
        f"[IG {tcga_id[0]}] risk(x)={endpoint_preds[steps]:.6f} "
        f"risk(baseline)={endpoint_preds[0]:.6f} delta={delta_pred:.6f} "
        f"sum(IG)={attribution_sum:.6f} completeness_error={completeness_error:.6f} "
        f"({100.0 * abs(completeness_error) / denom:.2f}% of delta)"
    )

    return {
        "integrated_grads": integrated_grads,
        "path_gradients": path_gradients,
        "vanilla_gradients": gradients[-1],
        "x_omic": x_np,
        "baseline": baseline_np,
        "completeness": {
            "tcga_id": tcga_id[0],
            "risk_input": endpoint_preds[steps],
            "risk_baseline": endpoint_preds[0],
            "delta_risk": delta_pred,
            "sum_ig": attribution_sum,
            "completeness_error": completeness_error,
            "rel_completeness_error": completeness_error / denom,
            "steps": steps,
            "baseline_kind": baseline_kind,
        },
    }


def run_inference(model, x_wsi, x_omic, return_attention=False):
    outputs_tuple = model(
        x_wsi=x_wsi,
        x_omic=x_omic,
        return_attention=return_attention,
    )
    if return_attention:
        outputs, wsi_embedding, omic_embedding, _, attention_weights = outputs_tuple
        attention_scores = attention_weights[0].squeeze().detach().cpu().numpy()
    else:
        outputs, wsi_embedding, omic_embedding, _ = outputs_tuple
        attention_scores = None

    return outputs, wsi_embedding, omic_embedding, attention_scores


def load_saved_attention_scores(tcga_id, attention_scores_dir):
    save_path = Path(attention_scores_dir) / f"attention_scores_{tcga_id[0]}.npy"
    if not save_path.exists():
        raise FileNotFoundError(f"Saved attention scores not found: {save_path}")
    return np.load(save_path)


def get_saliency_subset_indices(config, tcga_id, x_wsi, x_omic, model):
    selection = config.testing.saliency_tile_selection
    num_tiles = x_wsi.shape[1]

    if selection == "all":
        return np.arange(num_tiles)

    raise ValueError(f"Unsupported saliency tile selection: {selection}")


def subset_tiles(x_wsi, tile_names, tile_indices):
    tile_indices = np.asarray(tile_indices, dtype=np.int64)
    x_wsi_subset = x_wsi[:, tile_indices]
    if tile_names is None:
        tile_names_subset = None
    else:
        tile_names_subset = [tile_names[i] for i in tile_indices.tolist()]
    return x_wsi_subset, tile_names_subset


def plot_saliency_maps(
    saliency,
    image,
    tcga_id,
    tile_name,
    patch_id,
    output_dir,
    threshold=0.8,
):
    saliency = saliency.detach().cpu().numpy()
    image = image.permute(1, 2, 0).detach().cpu().numpy()
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

    tile_stem = (
        Path(tile_name).stem if tile_name is not None else f"tile_{patch_id:04d}"
    )
    save_path = os.path.join(
        output_dir, f"saliency_overlay_{tcga_id[0]}_{tile_stem}.png"
    )
    plt.savefig(save_path)
    logger.info(f"saved saliency overlay to {save_path}")

    plt.close()


def interpret_omic():

    # not used since the code is already simple enough

    pass


def interpret_wsi(x_wsi, tcga_id, output_dir_saliency, tile_names=None, max_patches=10):
    if x_wsi.grad is None:
        raise RuntimeError("Gradients have not been computed for the WSI tiles.")

    patient_tiles = x_wsi[0]
    patient_grads = x_wsi.grad[0]
    saliency_maps = patient_grads.detach().abs().amax(dim=1)
    saliency_scores = saliency_maps.mean(dim=(1, 2)).cpu().numpy()

    logger.info("OBTAINING SALIENCY MAPS")
    for patch_idx in range(min(patient_tiles.shape[0], max_patches)):
        logger.info(
            f"Generating saliency map for patch index {patch_idx} out of {max_patches}"
        )
        tile_name = tile_names[patch_idx] if tile_names is not None else None
        plot_saliency_maps(
            saliency_maps[patch_idx],
            patient_tiles[patch_idx],
            tcga_id,
            tile_name,
            patch_idx,
            output_dir_saliency,
        )

    return saliency_scores


def build_tile_metadata(tile_names):
    if tile_names is None:
        return None

    xs = []
    ys = []
    for tile_name in tile_names:
        x_coord, y_coord = parse_tile_coordinates(tile_name)
        xs.append(x_coord)
        ys.append(y_coord)

    return {
        "tile_names": np.asarray(tile_names, dtype=object),
        "x_coords": np.asarray(xs, dtype=np.int32),
        "y_coords": np.asarray(ys, dtype=np.int32),
    }


def save_tile_scores(tcga_id, tile_names, output_dir, score_name, scores):
    payload = build_tile_metadata(tile_names)
    if payload is None or scores is None:
        return

    payload[score_name] = np.asarray(scores, dtype=np.float32)

    save_path = os.path.join(output_dir, f"{score_name}_{tcga_id[0]}.npz")
    np.savez_compressed(save_path, **payload)
    logger.info(f"Saved tile-level attribution metadata to {save_path}")


def export_attention_maps(tcga_id, attention_scores, output_dir):
    if attention_scores is None:
        return

    save_path = os.path.join(output_dir, f"attention_scores_{tcga_id[0]}.npy")
    np.save(save_path, attention_scores)
    logger.info(f"Saved attention scores for {tcga_id[0]} to {save_path}")


def export_saliency_maps(
    config,
    model,
    device,
    tcga_id,
    x_wsi,
    x_omic,
    output_dir,
    tile_names,
    return_attention=False,
):
    max_saved = min(config.testing.saliency_max_tiles, x_wsi.shape[1])
    x_wsi = x_wsi.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        outputs, _, _, attention_scores = run_inference(
            model,
            x_wsi,
            x_omic,
            return_attention=return_attention,
        )

        if device.type == "cuda":
            allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
            reserved_memory = torch.cuda.memory_reserved(device) / (1024**3)

            logger.info(f"Allocated memory: {allocated_memory:.2f} GB")
            logger.info(f"Reserved memory: {reserved_memory:.2f} GB")
            logger.info(
                f"Free memory: {torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)} bytes"
            )
            torch.cuda.empty_cache()

        outputs.sum().backward()
        saliency_scores = interpret_wsi(
            x_wsi,
            tcga_id,
            output_dir,
            tile_names=tile_names,
            max_patches=max_saved,
        )

    return outputs.detach(), saliency_scores, attention_scores


def compute_saliency_chunked(model, x_wsi, x_omic, chunk_size):
    r"""Full-context per-tile saliency over ALL tiles, identical to the
    all-at-once backward (up to floating-point reduction order), but with memory
    bounded by ``chunk_size`` tiles instead of all :math:`T` at once.

    ----------------------------------------------------------------------------
    Model. Let the slide risk score be
    .. math::
        y \;=\; f\big(E_1, \dots, E_T\big), \qquad E_i \;=\; \mathrm{enc}(x_i),
    where :math:`x_i\in\mathbb{R}^{3\times H\times W}` is tile :math:`i`,
    :math:`\mathrm{enc}` is the per-tile foundation-model encoder, and
    :math:`E_i\in\mathbb{R}^{d}` is its embedding. Crucially :math:`E_i` depends
    on :math:`x_i` ALONE (tiles are encoded independently); the ONLY cross-tile
    interaction -- the mean query and the softmax over all tiles in the attention
    pool -- lives inside :math:`f` (= pool + fusion + MLP head).

    Quantity of interest. The per-tile saliency is the input gradient
    .. math::
        \frac{\partial y}{\partial x_i}
            \;=\; \underbrace{\frac{\partial y}{\partial E_i}}_{\textstyle g_i\in\mathbb{R}^{d}}
                  \;\underbrace{\frac{\partial E_i}{\partial x_i}}_{\textstyle J_i\;(\text{Jacobian of enc})}
            \;=\; g_i\,J_i ,
    by the chain rule. We then reduce :math:`\partial y/\partial x_i` to one
    scalar per tile (max over channels, mean over space), as in interpret_wsi.

    Why one backward OOMs. Computing :math:`\partial y/\partial x` in a single
    ``y.backward()`` forces autograd to keep the encoder graph (all activations)
    for every tile alive at once -- :math:`O(T)` activations of a 24-layer ViT.

    ----------------------------------------------------------------------------
    Stage 1 -- upstream gradient, full attention context, cheap.
    Treat the embeddings :math:`E` as leaves and backprop ONLY through :math:`f`:
    .. math::
        g \;=\; \nabla_E\, f(E_1,\dots,E_T)\Big|_{E=\mathrm{enc}(x)}
        \quad\Longrightarrow\quad
        g_i \;=\; \frac{\partial y}{\partial E_i}.
    No encoder activations are stored, and because :math:`f` is evaluated on ALL
    :math:`T` embeddings, each :math:`g_i` carries the global (all-tile) mean
    query and softmax -- i.e. the TRUE full-bag attention context.

    Stage 2 -- per-tile Jacobian, chunked.
    For a block of tiles :math:`C`, recompute :math:`E_C=\mathrm{enc}(x_C)` WITH a
    graph and seed the backward with the upstream gradient. For a vector-output
    node, ``E_C.backward(g_C)`` injects :math:`g_C` as the cotangent, computing
    the vector-Jacobian product
    .. math::
        \frac{\partial}{\partial x_i}\Big(g_i^\top E_i\Big)
            \;=\; g_i^\top \frac{\partial E_i}{\partial x_i}
            \;=\; g_i\,J_i
            \;=\; \frac{\partial y}{\partial x_i}, \qquad i\in C .
    The middle equality is exactly the chain-rule identity above, and there is no
    cross-tile leakage because :math:`E_i` depends only on :math:`x_i` (so
    :math:`\partial E_j/\partial x_i = 0` for :math:`j\neq i`). Only
    :math:`O(|C|)` encoder graphs are alive at a time.

    Exactness. Concatenating the stage-2 results over a partition of
    :math:`\{1,\dots,T\}` reconstructs :math:`\partial y/\partial x` exactly:
    .. math::
        \big\{\, g_i J_i \,\big\}_{i=1}^{T}
        \;=\; \frac{\partial y}{\partial x},
    so the chunked saliency EQUALS the all-at-once saliency (differences are only
    floating-point reduction order). The model weights and the map :math:`f` are
    never modified.

    Returns (output, saliency_scores[T], attention_scores), where each tile score
    uses interpret_wsi's reduction: max over channels then mean over space.
    """
    # ---- Stage 1: g_i = dy/dE_i with the FULL attention context ----
    with torch.no_grad():
        tile_features = model.encode_wsi_tile_features(x_wsi)  # [T, d_enc], detached
    embeddings_leaf = tile_features.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        output, _, _, _ = model.forward_from_wsi_tile_features(embeddings_leaf, x_omic)
        output.sum().backward()
    upstream_grad = embeddings_leaf.grad  # g, shape [T, d_enc]

    attention_scores = None
    if model.wsi_net.last_attention_weights is not None:
        attention_scores = (
            model.wsi_net.last_attention_weights.squeeze().detach().cpu().numpy()
        )

    # ---- Stage 2: vector-Jacobian product g_i^T (dE_i/dx_i), one chunk at a time ----
    patient_tiles = model._normalize_x_wsi_patients(x_wsi)[0]  # [T, 3, H, W]
    num_tiles = patient_tiles.shape[0]
    saliency_scores = torch.zeros(num_tiles)

    for start in range(0, num_tiles, chunk_size):
        end = min(start + chunk_size, num_tiles)
        tiles_chunk = patient_tiles[start:end].detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            features_chunk = model.encode_wsi_tile_features(tiles_chunk)  # [|C|, d_enc]
            # seed backward with g_C -> dy/dx_i = g_i^T dE_i/dx_i for i in C
            features_chunk.backward(upstream_grad[start:end])
        grads = tiles_chunk.grad  # [|C|, 3, H, W]
        saliency_scores[start:end] = (
            grads.abs().amax(dim=1).mean(dim=(1, 2)).detach().cpu()
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(
        "Computed chunked full-context saliency for %d tiles (chunk_size=%d)",
        num_tiles,
        chunk_size,
    )
    return output.detach(), saliency_scores.numpy(), attention_scores


def self_check_chunked_saliency(model, x_wsi, x_omic, n_tiles=8, atol=1e-4):
    r"""Validate the chunked path on the REAL model using a small sub-bag.

    Two identities (both proven exact above) are checked numerically:
      (i)  forward identity:
           :math:`f(\mathrm{enc}(x)) = \texttt{forward}(x)`, i.e.
           forward_from_wsi_tile_features(encode_wsi_tile_features(x)) == forward(x).
      (ii) saliency identity:
           chunked saliency == all-at-once saliency on the SAME bag.

    A small ``n_tiles`` is used so the all-at-once reference fits in memory; the
    chain-rule identity holds for any bag size, so this validates the method.
    """
    x_small = x_wsi[:, :n_tiles].contiguous()

    # (i) forward identity
    with torch.no_grad():
        out_ref, _, _, _ = model(x_wsi=x_small, x_omic=x_omic)
        feats = model.encode_wsi_tile_features(x_small)
        out_dec, _, _, _ = model.forward_from_wsi_tile_features(feats, x_omic)
    fwd_diff = float((out_ref - out_dec).abs().max().item())

    # (ii) all-at-once reference saliency on the same small bag
    x = x_small.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        outs, _, _, _ = run_inference(model, x, x_omic)
        outs.sum().backward()
    sal_full = x.grad[0].abs().amax(dim=1).mean(dim=(1, 2)).detach().cpu().numpy()

    # chunked saliency on the same small bag
    _, sal_chunk, _ = compute_saliency_chunked(
        model, x_small, x_omic, chunk_size=max(1, n_tiles // 2)
    )
    sal_diff = float(np.abs(sal_full - sal_chunk).max())

    passed = (fwd_diff <= atol) and (sal_diff <= atol)
    logger.info(
        "[self-check] forward max|Δ|=%.2e  saliency max|Δ|=%.2e  (atol=%.0e)  -> %s",
        fwd_diff,
        sal_diff,
        atol,
        "PASS" if passed else "FAIL",
    )
    if not passed:
        logger.warning(
            "[self-check] chunked saliency does NOT match the all-at-once reference"
        )
    return passed


def export_integrated_gradients(
    model, config, tcga_id, x_omic, x_wsi, baseline, output_dir
):
    steps = getattr(config.testing, "ig_steps", 50)
    result = calc_integrated_gradients(
        model, config, tcga_id, x_omic, x_wsi, baseline=baseline, steps=steps
    )

    tid = tcga_id[0]
    for prefix, key in (
        ("integrated_grads", "integrated_grads"),
        ("path_gradients", "path_gradients"),
        ("vanilla_gradients", "vanilla_gradients"),
        ("omic_input", "x_omic"),
    ):
        np.save(os.path.join(output_dir, f"{prefix}_{tid}.npy"), result[key])

    baseline_path = os.path.join(output_dir, "ig_baseline.npy")
    if not os.path.exists(baseline_path):
        np.save(baseline_path, result["baseline"])

    logger.info(f"Saved IG attributions + gradients for {tid} to {output_dir}")
    return result["completeness"]


def export_ig_completeness(records, output_dir):
    """Write the per-patient completeness check for the IG run.

    sum(IG) should equal risk(x) - risk(baseline). A large relative error means the
    Riemann sum is too coarse (raise testing.ig_steps) and the attributions -- and
    therefore every pathway score derived from them -- are not to be trusted.
    """
    if not records:
        return None
    import pandas as pd

    df = pd.DataFrame(records)
    path = os.path.join(output_dir, "ig_completeness.csv")
    df.to_csv(path, index=False)
    worst = df["rel_completeness_error"].abs().max()
    median = df["rel_completeness_error"].abs().median()
    logger.info(
        f"IG completeness over {len(df)} patients: median |error| "
        f"{100 * median:.2f}% of delta-risk, worst {100 * worst:.2f}% -> {path}"
    )
    if worst > 0.05:
        logger.warning(
            "IG completeness error exceeds 5% for at least one patient; consider "
            "raising testing.ig_steps."
        )
    return path


def export_attention_tile_scores(tcga_id, tile_names, output_dir, attention_scores):
    save_tile_scores(
        tcga_id,
        tile_names,
        output_dir,
        "attention_scores",
        attention_scores,
    )


def export_saliency_tile_scores(tcga_id, tile_names, output_dir, saliency_scores):
    save_tile_scores(
        tcga_id,
        tile_names,
        output_dir,
        "saliency_scores",
        saliency_scores,
    )


def test_and_interpret(config, model, test_loader, device, baseline=None):
    model.eval()
    all_tcga_ids = []
    all_predictions = []
    all_times = []
    all_events = []

    base_path = Path(config.testing.output_base_dir)
    output_dirs = setup_output_dirs(base_path)

    excluded_ids = [
        "TCGA-05-4395",
        "TCGA-86-8281",
    ]
    need_saliency = config.testing.calc_saliency_maps
    need_ig = config.testing.calc_IG
    need_attention = config.testing.calc_attention_maps
    validate_attribution_config(config)
    ig_completeness_records = []

    for batch_idx, batch in enumerate(test_loader):
        tcga_id, days_to_event, event_occurred, x_wsi, x_omic, tile_names = (
            unpack_test_batch(batch)
        )
        if tcga_id[0] in excluded_ids:
            logger.info(f"Skipping TCGA ID: {tcga_id}")
            continue

        x_wsi, x_omic, days_to_event, event_occurred = move_batch_to_device(
            x_wsi, x_omic, days_to_event, event_occurred, device
        )

        attention_scores = None
        saliency_scores = None
        saliency_tile_names = tile_names
        outputs = None

        need_full_slide_inference = (
            not need_saliency
            or need_attention
            or config.testing.saliency_tile_selection != "all"
        )

        if need_full_slide_inference:
            with torch.no_grad():
                outputs, _, _, attention_scores = run_inference(
                    model,
                    x_wsi,
                    x_omic,
                    return_attention=need_attention,
                )

        if need_saliency:
            if config.testing.saliency_chunked:
                # Full-context saliency over ALL tiles, memory-bounded. Identical
                # to the all-at-once backward (see compute_saliency_chunked).
                if config.testing.saliency_self_check and batch_idx == 0:
                    self_check_chunked_saliency(model, x_wsi, x_omic)
                saliency_outputs, saliency_scores, saliency_attention_scores = (
                    compute_saliency_chunked(
                        model,
                        x_wsi,
                        x_omic,
                        chunk_size=config.testing.saliency_chunk_size,
                    )
                )
                saliency_tile_names = tile_names
            else:
                saliency_tile_indices = get_saliency_subset_indices(
                    config,
                    tcga_id,
                    x_wsi,
                    x_omic,
                    model,
                )
                x_wsi_saliency, saliency_tile_names = subset_tiles(
                    x_wsi,
                    tile_names,
                    saliency_tile_indices,
                )
                saliency_outputs, saliency_scores, saliency_attention_scores = (
                    export_saliency_maps(
                        config,
                        model,
                        device,
                        tcga_id,
                        x_wsi_saliency,
                        x_omic,
                        output_dirs["saliency"],
                        saliency_tile_names,
                        return_attention=need_attention,
                    )
                )
            if outputs is None:
                outputs = saliency_outputs
            if attention_scores is None:
                attention_scores = saliency_attention_scores
            export_saliency_tile_scores(
                tcga_id,
                saliency_tile_names,
                output_dirs["saliency_tile_scores"],
                saliency_scores,
            )

        if need_attention:
            export_attention_maps(tcga_id, attention_scores, output_dirs["attention"])
            export_attention_tile_scores(
                tcga_id,
                tile_names,
                output_dirs["attention_tile_scores"],
                attention_scores,
            )

        if need_ig:
            ig_completeness_records.append(
                export_integrated_gradients(
                    model,
                    config,
                    tcga_id,
                    x_omic,
                    x_wsi,
                    baseline,
                    output_dirs["ig"],
                )
            )

        all_predictions.append(outputs.squeeze().detach().cpu().numpy())

        del outputs
        torch.cuda.empty_cache()
        all_tcga_ids.append(tcga_id)
        all_times.append(days_to_event)
        all_events.append(event_occurred)
        model.zero_grad()
        torch.cuda.empty_cache()

    if need_ig:
        export_ig_completeness(ig_completeness_records, output_dirs["ig"])

    all_predictions_np = [pred.item() for pred in all_predictions]
    all_events_np = torch.stack(all_events).cpu().numpy()
    all_events_bool_np = all_events_np.astype(bool)
    all_times_np = torch.stack(all_times).cpu().numpy()

    c_index = concordance_index_censored(
        all_events_bool_np.ravel(), all_times_np.ravel(), all_predictions_np
    )

    logger.info(f"CI: {c_index[0]}")

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
    logger.info(f"Log-Rank Test p-value: {p_value}")
    logger.info(f"Log-Rank Test statistic: {log_rank_results.test_statistic}")

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
