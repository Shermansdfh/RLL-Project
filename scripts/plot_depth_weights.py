"""
Plot learned depth-wise feature weighting from a checkpoint.

Usage:
    python scripts/plot_depth_weights.py --checkpoint path/to/action_head--XXXX_checkpoint.pt
    python scripts/plot_depth_weights.py --checkpoint path/to/action_head--XXXX_checkpoint.pt --save_dir figures/
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_weights(checkpoint_path: str):
    """Load kv_weight_logits and aq_weight_logits from an action head checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Try both prefixed and unprefixed keys
    kv_key = None
    aq_key = None
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

    kv_logits = state_dict[kv_key]  # (num_action_layers, 25) or (1, 25)
    aq_logits = state_dict[aq_key]

    # Convert logits -> softmax weights
    kv_weights = torch.softmax(kv_logits, dim=-1).numpy()  # (W, 25)
    aq_weights = torch.softmax(aq_logits, dim=-1).numpy()

    print(f"Loaded from: {checkpoint_path}")
    print(f"  kv_weight_logits shape: {kv_logits.shape}")
    print(f"  aq_weight_logits shape: {aq_logits.shape}")
    shared = kv_logits.shape[0] == 1
    print(f"  Weights shared across layers: {shared}")

    return kv_weights, aq_weights


def classify_layers(num_action_layers: int):
    """Split action head layers into shallow / mid / deep thirds."""
    indices = np.arange(num_action_layers)
    cuts = np.array_split(indices, 3)
    return {
        "Shallow": cuts[0],
        "Mid": cuts[1],
        "Deep": cuts[2],
    }


def plot_weights(weights: np.ndarray, title: str, ax: plt.Axes, num_vlm_layers: int):
    """
    Plot heatmap + average curves for shallow/mid/deep action head layers.

    weights: (num_action_layers, num_vlm_layers)
    """
    num_action_layers = weights.shape[0]
    vlm_layer_labels = ["emb"] + [str(i) for i in range(1, num_vlm_layers)]

    if num_action_layers == 1:
        # Shared weights — just plot a single bar chart
        ax.bar(range(num_vlm_layers), weights[0], color="#4C72B0", alpha=0.85)
        ax.set_xticks(range(num_vlm_layers))
        ax.set_xticklabels(vlm_layer_labels, fontsize=7, rotation=45)
        ax.set_xlabel("VLM Layer")
        ax.set_ylabel("Softmax Weight")
        ax.set_title(title)
        ax.axhline(1.0 / num_vlm_layers, color="gray", ls="--", lw=0.8, label="uniform")
        ax.legend(fontsize=8)
        return

    groups = classify_layers(num_action_layers)
    colors = {"Shallow": "#4C72B0", "Mid": "#DD8452", "Deep": "#55A868"}
    x = np.arange(num_vlm_layers)

    for group_name, layer_indices in groups.items():
        group_weights = weights[layer_indices]  # (subset, 25)
        mean = group_weights.mean(axis=0)
        std = group_weights.std(axis=0)
        ax.plot(x, mean, label=f"{group_name} (layers {layer_indices[0]}-{layer_indices[-1]})",
                color=colors[group_name], lw=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=colors[group_name])

    ax.axhline(1.0 / num_vlm_layers, color="gray", ls="--", lw=0.8, label="uniform (1/25)")
    ax.set_xticks(x)
    ax.set_xticklabels(vlm_layer_labels, fontsize=7, rotation=45)
    ax.set_xlabel("VLM Layer")
    ax.set_ylabel("Softmax Weight")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot depth-wise feature weighting from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to action_head--*_checkpoint.pt")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save figures (default: show interactively)")
    args = parser.parse_args()

    kv_weights, aq_weights = load_weights(args.checkpoint)
    num_vlm_layers = kv_weights.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    plot_weights(kv_weights, "Raw KV (Task Features) — Learned Depth Weights", axes[0], num_vlm_layers)
    plot_weights(aq_weights, "ActionQueries (Action Features) — Learned Depth Weights", axes[1], num_vlm_layers)

    fig.suptitle("Depth-Wise Feature Weighting Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "depth_wise_weights.png", dpi=150, bbox_inches="tight")
        print(f"Saved to {save_dir / 'depth_wise_weights.png'}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
