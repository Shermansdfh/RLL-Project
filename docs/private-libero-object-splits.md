# Private LIBERO-Object Splits

This document describes the private `libero_object` split support added for fine-tuning and rollout evaluation.

## Summary

Two shared split presets are supported through `--libero_object_private_split`:

- `stage1`
  - task IDs: `[0, 1, 2]`
  - training demos: `40` per task
  - validation demos: `10` per task
- `stage2`
  - task IDs: all `10` tasks in `libero_object`
  - training demos: `40` per task
  - validation demos: `10` per task

These presets are consumed by both:

- [`vla-scripts/finetune.py`](../vla-scripts/finetune.py)
- [`experiments/robot/libero/run_libero_eval.py`](../experiments/robot/libero/run_libero_eval.py)

The split definition is shared in:

- [`experiments/robot/libero/private_object_split.py`](../experiments/robot/libero/private_object_split.py)

## What Changed

### Shared split definition

Added a small shared module that defines:

- Stage 1 and Stage 2 split presets
- task-suite and dataset validation
- task ID to task language resolution
- RLDS selection metadata used by training

### Fine-tuning

[`vla-scripts/finetune.py`](../vla-scripts/finetune.py) now supports:

- `--libero_object_private_split {stage1|stage2}`
- `--dry_run True`

When a private split is provided:

- `--dataset_name` must be `libero_object_no_noops`
- training uses the shared Stage 1 or Stage 2 task selection
- validation uses the matching Stage 1 or Stage 2 selection

### RLDS dataset filtering

The RLDS loader now accepts deterministic trajectory selection metadata and filters trajectories by:

- selected task language
- capped demo count per selected task

This is implemented in:

- [`prismatic/vla/datasets/datasets.py`](../prismatic/vla/datasets/datasets.py)
- [`prismatic/vla/datasets/rlds/dataset.py`](../prismatic/vla/datasets/rlds/dataset.py)

### Rollout evaluation

[`experiments/robot/libero/run_libero_eval.py`](../experiments/robot/libero/run_libero_eval.py) now supports:

- `--libero_object_private_split {stage1|stage2}`
- `--dry_run True`

When a private split is provided:

- `--task_suite_name` must be `libero_object`
- rollout evaluation runs only on the matching Stage 1 or Stage 2 task IDs

### Tests

Added split logic coverage in:

- [`tests/test_private_libero_object_split.py`](../tests/test_private_libero_object_split.py)

## Usage

### Shared requirements

- Install dependencies from the repo README.
- Install LIBERO if you want to resolve task metadata or run rollout evaluation.
- Use `--dataset_name libero_object_no_noops` for fine-tuning.
- Use `--task_suite_name libero_object` for rollout evaluation.

## Stage 1

### Train

```bash
current_time=$(date +%Y%m%d-%H%M%S)
data_name=libero_object_no_noops

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name $data_name \
  --libero_object_private_split stage1 \
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
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "LIBERO-Object-Private-Stage1" \
  --run_id_note LIBERO-Object-Private-Stage1--train--$current_time
```

### Validation

```bash
current_time=$(date +%Y%m%d-%H%M%S)
data_name=libero_object_no_noops

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name $data_name \
  --libero_object_private_split stage1 \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --use_val_set True \
  --val_freq 5000 \
  --val_time_limit 180 \
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
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "LIBERO-Object-Private-Stage1" \
  --run_id_note LIBERO-Object-Private-Stage1--train-val--$current_time
```

### Rollout Eval

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Object-Private-Stage1 \
  --task_suite_name libero_object \
  --libero_object_private_split stage1 \
  --use_pro_version True
```

## Stage 2

### Train

```bash
current_time=$(date +%Y%m%d-%H%M%S)
data_name=libero_object_no_noops

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name $data_name \
  --libero_object_private_split stage2 \
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
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "LIBERO-Object-Private-Stage2" \
  --run_id_note LIBERO-Object-Private-Stage2--train--$current_time
```

### Validation

```bash
current_time=$(date +%Y%m%d-%H%M%S)
data_name=libero_object_no_noops

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name $data_name \
  --libero_object_private_split stage2 \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --use_val_set True \
  --val_freq 5000 \
  --val_time_limit 180 \
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
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "LIBERO-Object-Private-Stage2" \
  --run_id_note LIBERO-Object-Private-Stage2--train-val--$current_time
```

### Rollout Eval

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Object-Private-Stage2 \
  --task_suite_name libero_object \
  --libero_object_private_split stage2 \
  --use_pro_version True
```

## Dry-Run Commands

Use these to resolve the split and confirm task selection before running the full workflow.

### Stage 1 train dry run

```bash
python vla-scripts/finetune.py \
  --dataset_name libero_object_no_noops \
  --libero_object_private_split stage1 \
  --dry_run True
```

### Stage 1 eval dry run

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint outputs/LIBERO-Object-Private-Stage1 \
  --task_suite_name libero_object \
  --libero_object_private_split stage1 \
  --dry_run True
```

### Stage 2 train dry run

```bash
python vla-scripts/finetune.py \
  --dataset_name libero_object_no_noops \
  --libero_object_private_split stage2 \
  --dry_run True
```

### Stage 2 eval dry run

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint outputs/LIBERO-Object-Private-Stage2 \
  --task_suite_name libero_object \
  --libero_object_private_split stage2 \
  --dry_run True
```

## Smoke Tests Run

### Executed successfully

- `python -m pytest tests/test_private_libero_object_split.py`
- in-memory syntax compilation for all modified Python files
- split-resolution checks for:
  - Stage 1 task IDs
  - Stage 2 task IDs
  - Stage 1 train cap = `40` demos per selected task
  - Stage 1 val cap = `10` demos per selected task
  - Stage 2 train cap = `40` demos per selected task

### Verified from smoke tests

- Stage 1 resolves to task IDs `[0, 1, 2]`
- Stage 2 resolves to task IDs `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Stage 1 train selection caps at `40` demos per selected task
- Stage 1 val selection caps at `10` demos per selected task
- Stage 2 train selection caps at `40` demos per selected task

## Assumptions

- Validation remains part of the existing fine-tuning flow via `--use_val_set True`; there is no separate standalone validation entry point in this repo.
- The deterministic demo selection uses the first `N` trajectories encountered for each selected task language after forcing non-shuffled RLDS trajectory order before selection.
- Private split resolution requires the LIBERO benchmark package when task metadata must be resolved from the benchmark.
