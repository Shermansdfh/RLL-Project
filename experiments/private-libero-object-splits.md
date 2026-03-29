# Private LIBERO-Object Splits

Shared Stage 1 and Stage 2 presets are supported by:

- [`vla-scripts/finetune.py`](../vla-scripts/finetune.py)
- [`experiments/robot/libero/run_libero_eval.py`](robot/libero/run_libero_eval.py)

The shared split definition lives in:

- [`experiments/robot/libero/private_object_split.py`](robot/libero/private_object_split.py)

## Presets

| Preset | Task suite | Dataset | Task IDs | Train demos/task | Val demos/task |
| --- | --- | --- | --- | ---: | ---: |
| `stage1` | `libero_object` | `libero_object_no_noops` | `[0, 1, 2]` | 40 | 10 |
| `stage2` | `libero_object` | `libero_object_no_noops` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` | 40 | 10 |

## Implementation Notes

- Fine-tuning and rollout evaluation both consume the same split preset.
- RLDS trajectory filtering is deterministic and only applies to the targeted dataset.
- Selected LIBERO task languages must be unique. Duplicate task-language collisions are rejected.
- Validation remains part of the existing fine-tuning flow through `--use_val_set True`.

## Common Variables

```bash
current_time=$(date +%Y%m%d-%H%M%S)
data_name=libero_object_no_noops
split=stage1   # or stage2
project_name=LIBERO-Object-Private-${split}
checkpoint_dir=outputs/LIBERO-Object-Private-${split}
```

## Train

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
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "$project_name" \
  --run_id_note ${project_name}--train--$current_time \
  > logs/VLA-Adapter--${data_name}--${split}--${current_time}.log 2>&1 &
```

## Validation

```bash
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
  --wandb_project "$project_name" \
  --run_id_note ${project_name}--train-val--$current_time
```

## Rollout Eval

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint $checkpoint_dir \
  --task_suite_name libero_object \
  --libero_object_private_split $split \
  --use_pro_version True
```

## Dry Run

Use dry run to resolve the preset before launching the full workflow.

```bash
python vla-scripts/finetune.py \
  --dataset_name libero_object_no_noops \
  --libero_object_private_split $split \
  --dry_run True
```

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint $checkpoint_dir \
  --task_suite_name libero_object \
  --libero_object_private_split $split \
  --dry_run True
```

## Tests

Core split tests:

```bash
python -m pytest tests/test_private_libero_object_split.py
```

RLDS selection regression tests:

```bash
python -m pytest tests/test_rlds_trajectory_selection_characterization.py
```

## Verified Behavior

- `stage1` resolves to task IDs `[0, 1, 2]`
- `stage2` resolves to all 10 `libero_object` task IDs
- train filtering keeps exactly 40 demos per selected task
- val filtering keeps exactly 10 demos per selected task
- interleaved-mix selection logic is scoped to the targeted dataset
- selected task languages must be unique
