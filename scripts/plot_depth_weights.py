"""
Analyze learned depth-wise feature weighting from a checkpoint.

Computes routing metrics (variance, cosine similarity, entropy, top-k mass)
and produces diagnostic plots (heatmap, pairwise cosine, entropy per block).

Usage:
    python scripts/plot_depth_weights.py --checkpoint path/to/action_head--XXXX_checkpoint.pt
    python scripts/plot_depth_weights.py --checkpoint path/to/action_head--XXXX_checkpoint.pt --save_dir figures/
    python scripts/plot_depth_weights.py --checkpoint path/to/action_head--XXXX_checkpoint.pt --depth_curves
    python scripts/plot_depth_weights.py --checkpoint path/to/action_head--XXXX_checkpoint.pt --depth_curves --no_avg
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def load_weights(checkpoint_path: str):
    """Load kv_weight_logits and aq_weight_logits from an action head checkpoint.

    Returns softmax-normalized weights as numpy arrays, each (B, L) where
    B = number of action-head blocks, L = number of VLM layers.
    aq_weights is None if aq_weight_logits is not present (KV-only checkpoint).
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    kv_key = aq_key = None
    for k in state_dict.keys():
        if k.endswith("kv_weight_logits"):
            kv_key = k
        if k.endswith("aq_weight_logits"):
            aq_key = k

    if kv_key is None:
        all_keys = list(state_dict.keys())
        feature_keys = [k for k in all_keys if "feature_weighting" in k]
        raise KeyError(
            f"Could not find kv_weight_logits in checkpoint.\n"
            f"This checkpoint was likely trained without --use_depth_wise_weighting.\n"
            f"Keys containing 'feature_weighting': {feature_keys or '(none)'}\n"
            f"All keys ({len(all_keys)}): {all_keys}"
        )

    kv_logits = state_dict[kv_key]
    kv_weights = torch.softmax(kv_logits.float(), dim=-1).numpy()  # (B, L)

    aq_weights = None
    print(f"Loaded from: {checkpoint_path}")
    print(f"  kv_weight_logits shape: {kv_logits.shape}")

    if aq_key is not None:
        aq_logits = state_dict[aq_key]
        aq_weights = torch.softmax(aq_logits.float(), dim=-1).numpy()
        print(f"  aq_weight_logits shape: {aq_logits.shape}")
    else:
        print("  aq_weight_logits: not found (KV-only checkpoint)")

    shared = kv_logits.shape[0] == 1
    print(f"  Weights shared across blocks: {shared}")
    if shared:
        print("  (Metrics that compare across blocks are not meaningful for shared weights.)")

    return kv_weights, aq_weights


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────

def depth_variance(weights: np.ndarray) -> dict:
    """Measure how different the routing distributions are across blocks.

    Args:
        weights: (B, L) softmax routing weights.

    Returns:
        dict with per-layer variance and the averaged DepthVar scalar.
    """
    # Per-VLM-layer variance across blocks: Var over B for each l
    per_layer_var = weights.var(axis=0)          # (L,)
    depth_var = per_layer_var.mean()

    # Also: mean squared distance from global average
    w_global = weights.mean(axis=0, keepdims=True)  # (1, L)
    msd = np.mean(np.sum((weights - w_global) ** 2, axis=1))

    return {
        "per_layer_var": per_layer_var,
        "depth_var": float(depth_var),
        "mean_sq_dist_from_global": float(msd),
    }


def pairwise_cosine(weights: np.ndarray) -> dict:
    """Pairwise cosine similarity across blocks.

    Args:
        weights: (B, L).

    Returns:
        dict with the (B, B) cosine matrix, mean/min pairwise cosine,
        and a SpecializationIndex in [0, 1].
    """
    # Normalize rows
    norms = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-12
    normed = weights / norms
    cos_matrix = normed @ normed.T  # (B, B)

    B = weights.shape[0]
    # Upper-triangle (excluding diagonal)
    triu_idx = np.triu_indices(B, k=1)
    pairwise_vals = cos_matrix[triu_idx]

    mean_cos = float(pairwise_vals.mean()) if len(pairwise_vals) > 0 else 1.0
    min_cos = float(pairwise_vals.min()) if len(pairwise_vals) > 0 else 1.0
    specialization_index = 1.0 - mean_cos

    return {
        "cos_matrix": cos_matrix,
        "mean_pairwise_cosine": mean_cos,
        "min_pairwise_cosine": min_cos,
        "specialization_index": specialization_index,
    }


def pairwise_js_divergence(weights: np.ndarray) -> dict:
    """Pairwise Jensen-Shannon divergence across blocks.

    Args:
        weights: (B, L) — each row is a probability distribution over VLM layers.

    Returns:
        dict with (B, B) JS divergence matrix (in [0, 1] using log2),
        mean/max pairwise JSD.
    """
    B = weights.shape[0]
    w = np.clip(weights, 1e-12, None)

    js_matrix = np.zeros((B, B))
    for i in range(B):
        for j in range(i + 1, B):
            m = 0.5 * (w[i] + w[j])
            kl_im = np.sum(w[i] * np.log2(w[i] / m))
            kl_jm = np.sum(w[j] * np.log2(w[j] / m))
            jsd = 0.5 * (kl_im + kl_jm)
            js_matrix[i, j] = jsd
            js_matrix[j, i] = jsd

    triu_idx = np.triu_indices(B, k=1)
    pairwise_vals = js_matrix[triu_idx]

    mean_jsd = float(pairwise_vals.mean()) if len(pairwise_vals) > 0 else 0.0
    max_jsd = float(pairwise_vals.max()) if len(pairwise_vals) > 0 else 0.0

    return {
        "js_matrix": js_matrix,
        "mean_pairwise_jsd": mean_jsd,
        "max_pairwise_jsd": max_jsd,
    }


def entropy_metrics(weights: np.ndarray) -> dict:
    """Entropy and top-k mass per block.

    Args:
        weights: (B, L).

    Returns:
        dict with per-block entropy (raw and normalized), top-1 and top-3 mass.
    """
    L = weights.shape[1]
    log_L = np.log(L)

    # Clamp for log stability
    w = np.clip(weights, 1e-12, None)
    per_block_entropy = -np.sum(w * np.log(w), axis=1)       # (B,)
    per_block_entropy_norm = per_block_entropy / log_L        # (B,) in [0, 1]

    # Top-k mass
    sorted_w = np.sort(weights, axis=1)[:, ::-1]             # descending
    top1_mass = sorted_w[:, 0]                                # (B,)
    top3_mass = sorted_w[:, :3].sum(axis=1)                   # (B,)

    return {
        "per_block_entropy": per_block_entropy,
        "per_block_entropy_norm": per_block_entropy_norm,
        "mean_entropy_norm": float(per_block_entropy_norm.mean()),
        "top1_mass": top1_mass,
        "top3_mass": top3_mass,
        "mean_top1_mass": float(top1_mass.mean()),
        "mean_top3_mass": float(top3_mass.mean()),
    }


def compute_all_metrics(weights: np.ndarray, label: str) -> dict:
    """Compute and print all metrics for one set of weights."""
    dv = depth_variance(weights)
    pc = pairwise_cosine(weights)
    js = pairwise_js_divergence(weights)
    em = entropy_metrics(weights)

    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  DepthVar (mean per-layer var across blocks):  {dv['depth_var']:.6f}")
    print(f"  Mean squared dist from global avg:            {dv['mean_sq_dist_from_global']:.6f}")
    print(f"  Mean pairwise cosine similarity:              {pc['mean_pairwise_cosine']:.4f}")
    print(f"  Min  pairwise cosine similarity:              {pc['min_pairwise_cosine']:.4f}")
    print(f"  Specialization index (1 - mean cosine):       {pc['specialization_index']:.4f}")
    print(f"  Mean pairwise JS divergence:                  {js['mean_pairwise_jsd']:.6f}")
    print(f"  Max  pairwise JS divergence:                  {js['max_pairwise_jsd']:.6f}")
    print(f"  Mean normalized entropy (0=sharp, 1=uniform): {em['mean_entropy_norm']:.4f}")
    print(f"  Mean top-1 mass:                              {em['mean_top1_mass']:.4f}")
    print(f"  Mean top-3 mass:                              {em['mean_top3_mass']:.4f}")

    return {"depth_variance": dv, "pairwise_cosine": pc, "js_divergence": js, "entropy": em}


# ──────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────

def _vlm_layer_labels(L: int):
    return ["emb"] + [str(i) for i in range(1, L)]


def plot_routing_heatmap(weights: np.ndarray, title: str, ax: plt.Axes):
    """Plot 1: Heatmap of routing weights (blocks x VLM layers)."""
    B, L = weights.shape
    vlm_labels = _vlm_layer_labels(L)

    im = ax.imshow(weights, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("VLM Layer")
    ax.set_ylabel("DiT Block")
    ax.set_title(title)
    ax.set_xticks(range(L))
    ax.set_xticklabels(vlm_labels, fontsize=6, rotation=45)
    ax.set_yticks(range(B))
    ax.set_yticklabels(range(B), fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Weight")


def plot_cosine_heatmap(cos_matrix: np.ndarray, title: str, ax: plt.Axes):
    """Plot 2: Pairwise cosine similarity heatmap across blocks."""
    B = cos_matrix.shape[0]
    im = ax.imshow(cos_matrix, vmin=0, vmax=1, cmap="coolwarm", interpolation="nearest")
    ax.set_xlabel("DiT Block")
    ax.set_ylabel("DiT Block")
    ax.set_title(title)
    ax.set_xticks(range(B))
    ax.set_xticklabels(range(B), fontsize=6)
    ax.set_yticks(range(B))
    ax.set_yticklabels(range(B), fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine Sim")


def plot_entropy_per_block(entropy_norm: np.ndarray, top1: np.ndarray,
                           top3: np.ndarray, title: str, ax: plt.Axes):
    """Plot 3: Normalized entropy and top-k mass per block."""
    B = len(entropy_norm)
    x = np.arange(B)

    ax.bar(x - 0.2, entropy_norm, width=0.2, label="Norm. Entropy", color="#4C72B0", alpha=0.85)
    ax.bar(x, top1, width=0.2, label="Top-1 Mass", color="#DD8452", alpha=0.85)
    ax.bar(x + 0.2, top3, width=0.2, label="Top-3 Mass", color="#55A868", alpha=0.85)

    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("DiT Block")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(range(B), fontsize=6)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def _classify_layers(num_blocks: int):
    """Split action head blocks into shallow / mid / deep thirds."""
    indices = np.arange(num_blocks)
    cuts = np.array_split(indices, 3)
    return {"Shallow": cuts[0], "Mid": cuts[1], "Deep": cuts[2]}


def _pick_representative_layers(num_blocks: int):
    """Pick one shallow, one mid, one deep block index."""
    mid = num_blocks // 2
    return {
        f"Shallow (block 1)": 1,
        f"Mid (block {mid})": mid,
        f"Deep (block {num_blocks - 1})": num_blocks - 1,
    }


def plot_depth_curves(weights: np.ndarray, title: str, ax: plt.Axes, no_avg: bool = False):
    """Plot 4: Original shallow/mid/deep weight curves across VLM layers."""
    B, L = weights.shape
    vlm_labels = _vlm_layer_labels(L)
    x = np.arange(L)

    if B == 1:
        ax.bar(x, weights[0], color="#4C72B0", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(vlm_labels, fontsize=7, rotation=45)
        ax.set_xlabel("VLM Layer")
        ax.set_ylabel("Softmax Weight")
        ax.set_title(title)
        ax.axhline(1.0 / L, color="gray", ls="--", lw=0.8, label="uniform")
        ax.legend(fontsize=8)
        return

    colors_list = ["#4C72B0", "#DD8452", "#55A868"]

    if no_avg:
        representatives = _pick_representative_layers(B)
        for (label, idx), color in zip(representatives.items(), colors_list):
            ax.plot(x, weights[idx], label=label, color=color, lw=2, marker="o", markersize=3)
    else:
        groups = _classify_layers(B)
        for (group_name, block_indices), color in zip(groups.items(), colors_list):
            group_weights = weights[block_indices]
            mean = group_weights.mean(axis=0)
            std = group_weights.std(axis=0)
            ax.plot(x, mean, label=f"{group_name} (blocks {block_indices[0]}-{block_indices[-1]})",
                    color=color, lw=2)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    ax.axhline(1.0 / L, color="gray", ls="--", lw=0.8, label=f"uniform (1/{L})")
    ax.set_xticks(x)
    ax.set_xticklabels(vlm_labels, fontsize=7, rotation=45)
    ax.set_xlabel("VLM Layer")
    ax.set_ylabel("Softmax Weight")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_js_heatmap(js_matrix: np.ndarray, title: str, ax: plt.Axes):
    """Plot pairwise Jensen-Shannon divergence heatmap across blocks."""
    B = js_matrix.shape[0]
    vmax = js_matrix.max() if js_matrix.max() > 0 else 1.0
    im = ax.imshow(js_matrix, vmin=0, vmax=vmax, cmap="viridis", interpolation="nearest")
    ax.set_xlabel("DiT Block")
    ax.set_ylabel("DiT Block")
    ax.set_title(title)
    ax.set_xticks(range(B))
    ax.set_xticklabels(range(B), fontsize=6)
    ax.set_yticks(range(B))
    ax.set_yticklabels(range(B), fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="JS Divergence")


def make_plots(weights: np.ndarray, metrics: dict, feature_name: str,
               depth_curves: bool = False, no_avg: bool = False):
    """Create figure for one feature type (KV or AQ).

    Default 2x2: routing heatmap, cosine heatmap, JS heatmap, entropy/top-k.
    With --depth_curves: adds a 5th panel (3+2 layout).
    """
    B, L = weights.shape
    pc = metrics["pairwise_cosine"]
    js = metrics["js_divergence"]
    em = metrics["entropy"]

    if depth_curves:
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.4)
        ax_heatmap = fig.add_subplot(gs[0, 0:2])
        ax_cosine  = fig.add_subplot(gs[0, 2:4])
        ax_js      = fig.add_subplot(gs[0, 4:6])
        ax_entropy = fig.add_subplot(gs[1, 0:3])
        ax_curves  = fig.add_subplot(gs[1, 3:6])
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        ax_heatmap = axes[0, 0]
        ax_cosine  = axes[0, 1]
        ax_js      = axes[1, 0]
        ax_entropy = axes[1, 1]

    plot_routing_heatmap(weights, f"{feature_name} — Routing Weights", ax_heatmap)

    if B > 1:
        plot_cosine_heatmap(pc["cos_matrix"],
                            f"{feature_name} — Pairwise Cosine (spec={pc['specialization_index']:.3f})",
                            ax_cosine)
        plot_js_heatmap(js["js_matrix"],
                        f"{feature_name} — Pairwise JS Div (mean={js['mean_pairwise_jsd']:.4f})",
                        ax_js)
    else:
        for ax, name in [(ax_cosine, "Pairwise Cosine"), (ax_js, "Pairwise JS Div")]:
            ax.text(0.5, 0.5, "Shared weights\n(single block)",
                    ha="center", va="center", fontsize=14)
            ax.set_title(f"{feature_name} — {name}")

    plot_entropy_per_block(em["per_block_entropy_norm"], em["top1_mass"], em["top3_mass"],
                           f"{feature_name} — Entropy & Top-k Mass", ax_entropy)

    if depth_curves:
        plot_depth_curves(weights, f"{feature_name} — Depth Weight Curves", ax_curves, no_avg=no_avg)

    fig.suptitle(f"Depth-Wise Routing Analysis — {feature_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze depth-wise feature weighting from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to action_head--*_checkpoint.pt")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save figures (default: show interactively)")
    parser.add_argument("--depth_curves", action="store_true",
                        help="Also plot the original shallow/mid/deep depth weight curves")
    parser.add_argument("--no_avg", action="store_true",
                        help="With --depth_curves, plot individual representative blocks instead of group averages")
    args = parser.parse_args()

    kv_weights, aq_weights = load_weights(args.checkpoint)

    kv_metrics = compute_all_metrics(kv_weights, "KV (Task Features)")
    fig_kv = make_plots(kv_weights, kv_metrics, "KV", depth_curves=args.depth_curves, no_avg=args.no_avg)

    fig_aq = None
    if aq_weights is not None:
        aq_metrics = compute_all_metrics(aq_weights, "AQ (Action Queries)")
        fig_aq = make_plots(aq_weights, aq_metrics, "AQ", depth_curves=args.depth_curves, no_avg=args.no_avg)
    else:
        print("\nSkipping AQ analysis (KV-only checkpoint).")

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_kv.savefig(save_dir / "routing_analysis_kv.png", dpi=150, bbox_inches="tight")
        print(f"\nSaved to {save_dir / 'routing_analysis_kv.png'}")
        if fig_aq is not None:
            fig_aq.savefig(save_dir / "routing_analysis_aq.png", dpi=150, bbox_inches="tight")
            print(f"Saved to {save_dir / 'routing_analysis_aq.png'}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
