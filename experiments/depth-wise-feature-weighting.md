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
  | Softmax-weighted sum     |   (separate learnable weights for raw K/V and AQ)
  +--------------------------+
        |
        v
  Bridge Attention block i   (unchanged — only its inputs differ)
```

For each action-head block *i*:

1. **LayerNorm** is applied independently to each VLM layer's raw K/V features, and independently to each VLM layer's ActionQueries. Each VLM layer has its own learned scale and bias, so features from different depths are normalized to comparable scales.

2. **Learnable weights** (one set for raw K/V, one set for ActionQueries) are softmax-normalized and used to compute a weighted sum across VLM layers.

3. The resulting combined K/V and combined ActionQueries are fed into the Bridge Attention block, which is unchanged.

## Configuration

Three flags control the feature, all in `FinetuneConfig`:

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--use_depth_wise_weighting` | `bool` | `False` | Enable depth-wise feature weighting |
| `--share_depth_weights` | `bool` | `False` | If `True`, all 24 action-head blocks share one set of mixing weights. If `False`, each block learns its own. |
| `--normalize_aq_before_combination` | `bool` | `True` | If `True`, ActionQueries are LayerNorm-ed before the weighted sum. If `False`, raw (un-normalized) ActionQueries are combined. The LayerNorm modules are still created (for easy switching) but bypassed. |

When `--use_depth_wise_weighting False` (the default), behavior is identical to the original codebase.

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
- Weight logits are initialized to zero, so the initial softmax yields uniform 1/25 weights — a safe starting point equivalent to averaging all layers.
- Backward compatibility is preserved: when `--use_depth_wise_weighting False`, the code follows the original `h_t[:, i+1, :]` indexing path with zero overhead.
