# Depth-Wise Feature Weighting

Learnable linear combination of VLM hidden-state layers for Bridge Attention inputs.

Supported by:

- [`vla-scripts/finetune.py`](../vla-scripts/finetune.py)
- [`experiments/robot/openvla_utils.py`](robot/openvla_utils.py) (inference)

Implementation lives in:

- [`prismatic/models/action_heads.py`](../prismatic/models/action_heads.py) (`DepthWiseFeatureWeighting`, `MLPResNet`, `L1RegressionActionHead`)

Sanity-check test:

- [`tests/test_depth_wise_weighting.py`](../tests/test_depth_wise_weighting.py)

## Motivation

In the baseline VLA-Adapter, each action-head transformer block receives features from a single, fixed VLM layer (block *i* receives layer *i + 1*). This creates a rigid one-to-one mapping between VLM depth and action-head depth.

Depth-wise feature weighting replaces this with a **learnable linear combination** across all VLM layers, so each action-head block can freely attend to the VLM depth range that is most useful for its role.

## How It Works

```
VLM hidden states  (25 layers, from embedding through layer 24)
        |
        v
  +--------------------------+
  | Per-layer LayerNorm      |   (separate norms for raw K/V and ActionQueries)
  +--------------------------+
        |
        v
  +--------------------------+
  | Normalized weighted sum  |   (separate learnable weights for raw K/V and AQ; softmax/sparsemax/entmax15)
  +--------------------------+
        |
        v
  Bridge Attention block i   (unchanged — only its inputs differ)
```

For each action-head block *i*:

1. **LayerNorm** is applied independently to each VLM layer's raw K/V features, and independently to each VLM layer's ActionQueries. Each VLM layer has its own learned scale and bias, so features from different depths are normalized to comparable scales.

2. **Learnable weights** (one set for raw K/V, one set for ActionQueries) are normalized via the chosen activation (softmax, sparsemax, or entmax15) and used to compute a weighted sum across VLM layers.

3. The resulting combined K/V and combined ActionQueries are fed into the Bridge Attention block, which is unchanged.

## Configuration

Three flags control the feature, all in `FinetuneConfig`:

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--use_depth_wise_weighting` | `bool` | `False` | Enable depth-wise feature weighting |
| `--share_depth_weights` | `bool` | `False` | If `True`, all 24 action-head blocks share one set of mixing weights. If `False`, each block learns its own. |
| `--normalize_aq_before_combination` | `bool` | `True` | If `True`, ActionQueries are LayerNorm-ed before the weighted sum. If `False`, raw (un-normalized) ActionQueries are combined. The LayerNorm modules are still created (for easy switching) but bypassed. |
| `--depth_weight_top_k` | `int` | `0` | If `>0` and `< num_vlm_layers`, each action-head block mixes only its `k` highest-weighted VLM layers (others masked to zero via softmax `-inf`). `0` disables the gate (mix over all layers). |
| `--depth_weight_epsilon` | `float` | `0.0` | Epsilon-greedy exploration (training only). With probability ε, the top-k selection is replaced by a uniformly random draw of `k` VLM layers. Sampled independently per block and per stream (K/V vs AQ) on every forward pass. `0.0` disables exploration. Ignored at eval. |
| `--depth_weight_activation` | `str` | `"softmax"` | Activation used to normalize mixing logits. One of `"softmax"`, `"sparsemax"`, or `"entmax15"`. Sparsemax produces truly sparse weights (exact zeros), encouraging the model to focus on fewer VLM layers. `entmax15` (α=1.5) is a middle ground between softmax and sparsemax. Requires `pip install entmax`. |

When `--use_depth_wise_weighting False` (the default), behavior is identical to the original codebase.

### Top-K with Epsilon-Greedy Exploration

Setting `--depth_weight_top_k k` turns the dense softmax-over-all-layers into a sparse gate: only the `k` selected VLM layers contribute to the mix, with softmax re-normalized over the selected subset. Gradients flow only through the selected logits; the rest have zero gradient for that step.

`--depth_weight_epsilon ε` overrides the top-k choice with a random k-subset with probability ε during training. This keeps under-used VLM layers alive long enough to get gradient and potentially climb into the top-k. Use it when you observe the softmax collapsing to a narrow depth range early in training and want to encourage wider exploration.

Typical recipes:

- Dense mix (default): `--depth_weight_top_k 0`
- Sparse mix, no exploration: `--depth_weight_top_k 5 --depth_weight_epsilon 0.0`
- Sparse mix with exploration: `--depth_weight_top_k 5 --depth_weight_epsilon 0.1`

**Checkpoint compatibility:** `top_k` and `epsilon` change forward behavior, not parameter shapes, so checkpoints trained with one setting load cleanly under another. However, mixing weights learned under a sparse gate may look very different from dense-gate weights, so don't expect behavior to transfer.

## Parameter Budget

The module adds the following learnable parameters on top of the base action head:

| Component | Count | Note |
| --- | --- | --- |
| `kv_layer_norms` | 25 x LayerNorm(dim) | 25 VLM layers, each with scale + bias |
| `aq_layer_norms` | 25 x LayerNorm(dim) | Same, for ActionQueries |
| `kv_weight_logits` | *W* x 25 | *W* = 1 if shared, 24 otherwise |
| `aq_weight_logits` | *W* x 25 | Same |

With `dim = 4096` (production) and `share_depth_weights = False`: ~410 K additional parameters (negligible relative to the 207 MB Pro action head).

## Training Commands

All commands below assume the standard VLA-Adapter prerequisites are in place (conda env, data, pretrained VLM backbone). The depth-wise flags are simply appended to existing training commands.

### Common Variables

```bash
current_time=$(date +%Y%m%d-%H%M%S)
```

### Full Dataset (LIBERO-Object, all tasks and demos)

```bash
data_name=libero_object_no_noops

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name $data_name \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 400000 \
  --max_steps 400005 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --use_depth_wise_weighting True \
  --share_depth_weights False \
  --normalize_aq_before_combination True \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "LIBERO-Object-DW" \
  --run_id_note LIBERO-Object-DW--full--$current_time
```

Replace `libero_object_no_noops` with `libero_spatial_no_noops`, `libero_goal_no_noops`, or `libero_10_no_noops` for other LIBERO suites. For CALVIN, change `--data_root_dir data` and `--dataset_name calvin_abc`.

Adjust `--batch_size` and `--grad_accumulation_steps` to match your GPU VRAM (see [README.md](../README.md) for per-GPU guidance).

### Private Split — Stage 1 (3 tasks, fast iteration)

```bash
data_name=libero_object_no_noops
split=stage1
project_name=LIBERO-Object-DW-Private-${split}
current_time=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name $data_name \
  --libero_object_private_split $split \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 400000 \
  --max_steps 400005 \
  --save_freq 500 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 4 \
  --grad_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --use_depth_wise_weighting True \
  --share_depth_weights False \
  --normalize_aq_before_combination True \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "$project_name" \
  --run_id_note ${project_name}--train--$current_time \
  --seed 114 \
  > logs/VLA-Adapter--${data_name}--${split}--${current_time}.log 2>&1 &
```

### Private Split — Stage 2 (all 10 tasks)

Same command as Stage 1 but with `split=stage2`:

```bash
split=stage2
project_name=LIBERO-Object-DW-Private-${split}

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name $data_name \
  --libero_object_private_split $split \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 400000 \
  --max_steps 400005 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --use_depth_wise_weighting True \
  --share_depth_weights False \
  --normalize_aq_before_combination True \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "$project_name" \
  --run_id_note ${project_name}--train--$current_time
```

### With Validation Set

Append these flags to any of the commands above:

```bash
  --use_val_set True \
  --val_freq 5000 \
  --val_time_limit 180 \
```

### Dry Run (verify config without loading model)

```bash
python vla-scripts/finetune.py \
  --dataset_name libero_object_no_noops \
  --libero_object_private_split stage1 \
  --use_depth_wise_weighting True \
  --dry_run True
```

## Rollout Evaluation

Depth-wise weighting is transparent to the eval pipeline. The action head checkpoint contains the `DepthWiseFeatureWeighting` weights, so `run_libero_eval.py` loads them automatically via `state_dict`. Pass the same flags used during training:

```bash
split=stage1   # or stage2
checkpoint_dir=outputs/LIBERO-Object-DW-Private-${split}

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint $checkpoint_dir \
  --task_suite_name libero_object \
  --libero_object_private_split $split \
  --use_pro_version True \
  --use_depth_wise_weighting True \
  --share_depth_weights False \
  --normalize_aq_before_combination True
```

**Important:** The depth-wise flags must match between training and evaluation. If you trained with `--share_depth_weights True`, you must pass the same flag at eval time, otherwise the `state_dict` shapes will mismatch.

## Analyzing Learned Weights

[`scripts/plot_depth_weights.py`](../scripts/plot_depth_weights.py) loads a checkpoint and produces routing diagnostics (heatmap, pairwise cosine/JS divergence, entropy, depth curves). Works with both full and KV-only checkpoints (AQ analysis is skipped when `aq_weight_logits` is absent).

```bash
# Interactive display
python scripts/plot_depth_weights.py --checkpoint path/to/action_head--XXXX_checkpoint.pt

# Save figures
python scripts/plot_depth_weights.py --checkpoint path/to/action_head--XXXX_checkpoint.pt --save_dir figures/

# Include shallow/mid/deep weight curves (group averages ± std)
python scripts/plot_depth_weights.py --checkpoint path/to/action_head--XXXX_checkpoint.pt --depth_curves

# Plot individual representative blocks instead of group averages
python scripts/plot_depth_weights.py --checkpoint path/to/action_head--XXXX_checkpoint.pt --depth_curves --no_avg
```

## Sanity-Check Test

Run the offline test (no data or pretrained weights needed) to verify shape, gradient flow, device transfer, and a 2-batch smoke train:

```bash
python tests/test_depth_wise_weighting.py
```

Expected output: `23 passed, 0 failed`.

## Split Presets Reference

See [`experiments/private-libero-object-splits.md`](private-libero-object-splits.md) for the full split preset definitions.

| Preset | Tasks | Train demos/task | Val demos/task |
| --- | ---: | ---: | ---: |
| `stage1` | 3 (IDs 0-2) | 40 | 10 |
| `stage2` | 10 (IDs 0-9) | 40 | 10 |

## Design Notes

- Bridge Attention blocks (`MLPResNetBlock`, `MLPResNetBlock_Pro`) are completely unchanged. Only the construction of their `h_t` and `h_a` inputs is different.
- Weight logits are initialized to zero, so the initial activation yields uniform 1/25 weights — a safe starting point equivalent to averaging all layers.
- Backward compatibility is preserved: when `--use_depth_wise_weighting False`, the code follows the original `h_t[:, i+1, :]` indexing path with zero overhead.
- Top-k masking is implemented by adding `-inf` to unselected logits before the activation (not by slicing), so the combine step stays vectorized and gradient flow through selected logits is identical to the dense case.
- Epsilon-greedy draws use `torch.randperm` on the current device, one draw per (block × stream) per forward. It respects `model.train()` / `model.eval()` — at eval time only the deterministic top-k path runs.

## Related Ablation: Pure-KV-Cache Mode

Two additional flags turn off the two learnable components that sit between the VLM and the action head, reducing the architecture to a pure layer-wise cross-attention link against the VLM KV cache.

| Flag | Type | Default | Effect when `False` |
| --- | --- | --- | --- |
| `--use_action_queries` | `bool` | `True` | VLM no longer injects learnable `action_queries` at action-token positions — those positions are filled with fixed zero embeddings. The action head drops its `h_a` (adapter) cross-attention branch; the `action_queries.weight` parameter is frozen (`requires_grad = False`). |
| `--use_kv_gate` | `bool` | `True` | Bridge Attention blocks stop applying `tanh(gating_factor)` to the VLM-KV (task) cross-attention scores; the gate is fixed at `1.0`. The `gating_factor` parameter is still registered (for checkpoint compatibility) but unused. |

Setting both to `False` (and `--use_proprio False`) yields a decoder whose only cross-attention input is the VLM's per-layer KV cache — "pure layer-wise link of attention with VLM KV cache."

### What Gets Touched

| File | Change |
| --- | --- |
| [`vla-scripts/finetune.py`](../vla-scripts/finetune.py) | Adds `use_action_queries` / `use_kv_gate` to `FinetuneConfig`; forwards them to `L1RegressionActionHead`; sets `vla.use_action_queries` after `from_pretrained`; gates the `action_queries` `requires_grad` flip on `use_action_queries`. |
| [`prismatic/extern/hf/modeling_prismatic.py`](../prismatic/extern/hf/modeling_prismatic.py) | `PrismaticForConditionalGeneration.__init__` registers `self.use_action_queries = True`. In `forward()`, the action-position replacement branches: learnable `action_queries` when enabled, fixed zero embeddings when disabled. |
| [`prismatic/models/action_heads.py`](../prismatic/models/action_heads.py) | Both flags threaded through `L1RegressionActionHead` → `MLPResNet` → `MLPResNetBlock` / `MLPResNetBlock_Pro`. Blocks accept `h_a=None`, skip the adapter cross-attention branch when `use_action_queries=False`, and use `ratio_g = 1.0` when `use_kv_gate=False`. `DepthWiseFeatureWeighting.normalize_all` / `combine` tolerate `h_a_all=None`. `L1RegressionActionHead.predict_action` stops slicing out the action-position hidden states when `use_action_queries=False`. |

### Compatibility Matrix

| `use_action_queries` | `use_kv_gate` | Behavior |
| --- | --- | --- |
| `True` | `True` | Original VLA-Adapter architecture (default). |
| `True` | `False` | Action queries kept; gate fixed to 1. |
| `False` | `True` | Zero-embedded action positions; adapter branch dropped; gate still on VLM-KV scores. |
| `False` | `False` | Pure layer-wise link: self-attention + un-gated VLM-KV cross-attention (+ proprio, if enabled). |

### Training Example — Pure-KV-Cache

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name libero_object_no_noops \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora True \
  --use_minivlm True \
  --use_pro_version True \
  --use_action_queries False \
  --use_kv_gate False \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "LIBERO-Object-PureKV" \
  --run_id_note LIBERO-Object-PureKV--$(date +%Y%m%d-%H%M%S)
```

Combine with `--use_depth_wise_weighting True` if you also want learnable layer mixing on the VLM KV — the two features are orthogonal.

### Caveats

- **Checkpoints are not cross-compatible.** A pure-KV-cache checkpoint drops the projections for `h_a` in the Pro block (`k_adapter`, `v_adapter` are still registered but untrained in the adapter-dropped code path) and leaves `action_queries.weight` at zero. Loading such a checkpoint with `--use_action_queries True` will produce a silently broken model.
- **Inference script not yet wired.** `experiments/robot/openvla_utils.py` and `vla-scripts/vla_evaluation.py` currently load models without forwarding these flags — before running rollouts on a pure-KV checkpoint, set `vla.use_action_queries = False` manually and reconstruct the action head with matching flags.
- **Parameter budget.** When `use_action_queries=False`, the `action_queries` embedding (`NUM_TOKENS × llm_dim`) and the `k_adapter`/`v_adapter` projections in each Pro block are dead weight — still loaded, but never receive gradients or contribute to the forward pass.
