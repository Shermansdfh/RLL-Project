"""
Analyze learned depth-wise feature weighting from a checkpoint.

Computes routing metrics (variance, cosine similarity, entropy, top-k mass)
and produces diagnostic plots (heatmap, pairwise cosine, entropy per block).

Usage:
    python scripts/plot_depth_weights.py --checkpoint path/to/action_head--XXXX_checkpoint.pt
    python scripts/plot_depth_weights.py --checkpoint path/to/action_head--XXXX_checkpoint.pt --save_dir figures/
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
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    kv_key = aq_key = None
    for k in state_dict.keys():
        if k.endswith("kv_weight_logits"):
            kv_key = k
        if k.endswith("aq_weight_logits"):
            aq_key = k

    if kv_key is None or aq_key is None:
        raise KeyError(
            f"Could not find kv_weight_logits / aq_weight_logits in checkpoint.\n"
            f"Available keys (first 20): {list(state_dict.keys())[:20]}"
        )

    kv_logits = state_dict[kv_key]
    aq_logits = state_dict[aq_key]

    kv_weights = torch.softmax(kv_logits.float(), dim=-1).numpy()  # (B, L)
    aq_weights = torch.softmax(aq_logits.float(), dim=-1).numpy()

    print(f"Loaded from: {checkpoint_path}")
    print(f"  kv_weight_logits shape: {kv_logits.shape}")
    print(f"  aq_weight_logits shape: {aq_logits.shape}")
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
    em = entropy_metrics(weights)

    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  DepthVar (mean per-layer var across blocks):  {dv['depth_var']:.6f}")
    print(f"  Mean squared dist from global avg:            {dv['mean_sq_dist_from_global']:.6f}")
    print(f"  Mean pairwise cosine similarity:              {pc['mean_pairwise_cosine']:.4f}")
    print(f"  Min  pairwise cosine similarity:              {pc['min_pairwise_cosine']:.4f}")
    print(f"  Specialization index (1 - mean cosine):       {pc['specialization_index']:.4f}")
    print(f"  Mean normalized entropy (0=sharp, 1=uniform): {em['mean_entropy_norm']:.4f}")
    print(f"  Mean top-1 mass:                              {em['mean_top1_mass']:.4f}")
    print(f"  Mean top-3 mass:                              {em['mean_top3_mass']:.4f}")

    return {"depth_variance": dv, "pairwise_cosine": pc, "entropy": em}


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


def make_plots(weights: np.ndarray, metrics: dict, feature_name: str):
    """Create 3-panel figure for one feature type (KV or AQ)."""
    B, L = weights.shape
    pc = metrics["pairwise_cosine"]
    em = metrics["entropy"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_routing_heatmap(weights, f"{feature_name} — Routing Weights", axes[0])

    if B > 1:
        plot_cosine_heatmap(pc["cos_matrix"],
                            f"{feature_name} — Pairwise Cosine (spec={pc['specialization_index']:.3f})",
                            axes[1])
    else:
        axes[1].text(0.5, 0.5, "Shared weights\n(single block)",
                     ha="center", va="center", fontsize=14)
        axes[1].set_title(f"{feature_name} — Pairwise Cosine")

    plot_entropy_per_block(em["per_block_entropy_norm"], em["top1_mass"], em["top3_mass"],
                           f"{feature_name} — Entropy & Top-k Mass", axes[2])

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
    args = parser.parse_args()

    kv_weights, aq_weights = load_weights(args.checkpoint)

    kv_metrics = compute_all_metrics(kv_weights, "KV (Task Features)")
    aq_metrics = compute_all_metrics(aq_weights, "AQ (Action Queries)")

    fig_kv = make_plots(kv_weights, kv_metrics, "KV")
    fig_aq = make_plots(aq_weights, aq_metrics, "AQ")

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_kv.savefig(save_dir / "routing_analysis_kv.png", dpi=150, bbox_inches="tight")
        fig_aq.savefig(save_dir / "routing_analysis_aq.png", dpi=150, bbox_inches="tight")
        print(f"\nSaved to {save_dir / 'routing_analysis_kv.png'}")
        print(f"Saved to {save_dir / 'routing_analysis_aq.png'}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
