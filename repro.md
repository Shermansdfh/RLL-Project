
### Spatial
- 500 runs = 10 tasks × 50 episodes/task
- Total wall time: 57:44 = 3464 seconds
- Time per episode: 3464 / 500 = 6.928 s/episode
- CUDA / driver: Driver Version: 580.95.05 / CUDA Version: 13.0
- torch version: 2.2.0+cu121
- VRAM peak for LIBERO-Spatial is 44xx MB
- `huggingface-cli download VLA-Adapter/LIBERO-Spatial --local-dir outputs/LIBERO-Spatial`
### Long
Util around 50%, may be env/CPU bottlenecked

|Suite|ckpt|SR avg|sec/episode|total time|peak VRAM|Peak util|
|---|---|---|---|---|---|---|
|Spatial|LIBERO-Spatial|93.0%|6.93|57:44|~4.5GB|NaN|
|10	|LIBERO-Long|94.0%|12.194|1:41:37|~4.6GB|46%|
|Goal|Libero-Goal|96.0%|6.93|50:24|~4.3GB|41%|
|Object|Libero-Object|91.4%|6.93|1:00:27|~4.4GB|41%|

#### Reproducibility check (1 task, 10 episodes, same seed twice)
From repo root:
```bash
python -m experiments.robot.libero.repro_check_libero \
  --pretrained_checkpoint outputs/LIBERO-Long \
  --task_suite libero_10 --task_id 0 --num_episodes 10 --seed 7 --no_videos
```
Exits 0 if both runs have identical success rate; else exits 1 and prints diff.

#### Long suite bottleneck profiling (env vs model vs data)
From repo root:
```bash
# Time split: env.step vs model forward vs prepare_observation
python -m experiments.robot.libero.profile_libero_long \
  --pretrained_checkpoint outputs/LIBERO-Long --num_episodes 2

# Test if rendering is limiter (lower res)
python -m experiments.robot.libero.profile_libero_long \
  --pretrained_checkpoint outputs/LIBERO-Long --env_img_res 128 --num_episodes 2

# Run 4 episodes in parallel (increase env workers until CPU saturates)
python -m experiments.robot.libero.profile_libero_long \
  --pretrained_checkpoint outputs/LIBERO-Long --workers 4
```
Outcome: (A) faster eval via more workers / lower res, or (B) proof it's env-bound so you don't waste time on CUDA tricks.

### :microscope: Quick 1-Task Training & Evaluation (Small Steps)

***=> Show loss decreases and SR increases on a single task with minimal compute.***

Train on **1 task** (or 2) with small steps (1k → 5k → 10k), then evaluate only that task. Uses the [Extremely Limited VRAM](#ledger-how-to-train-on-extremely-limited-vram-gpus) config (batch_size=1, grad_accumulation_steps=8, etc.).

**1. Train (1 task, 1k steps):**
```bash
data_name=libero_spatial_no_noops
current_time=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False --num_images_in_input 2 --use_proprio True \
--use_lora True --use_fz False --use_minivlm True \
--image_aug True \
--batch_size 1 --grad_accumulation_steps 8 --learning_rate 2e-4 --lora_rank 64 \
--max_steps 1001 --save_freq 1000 --save_latest_checkpoint_only False \
--merge_lora_during_training True --use_pro_version True \
--libero_task_ids 0 \
--run_id_note 1task-1k--$current_time \
> logs/1task-1k--$current_time.log 2>&1 &
```

**2. Repeat for 5k and 10k** by changing `--max_steps 5001` / `--save_freq 5000` and `--max_steps 10001` / `--save_freq 10000`.

**2b. Resume from a checkpoint (e.g. keep training past 1k):**  
If you have a checkpoint at step 1000 and want to continue to 5k (or any higher step), use `--resume True`, `--resume_step 1000`, and point both `--config_file_path` and `--resum_vla_path` at that checkpoint directory. Set `--max_steps` to the new total (e.g. `5001`).

```bash
# Example: resume from outputs/<run_id>--1000_chkpt and train until step 5000
CHKPT_DIR=outputs/<your_run_id>--1000_chkpt   # e.g. outputs/prism-...+1task-1k--20260301_231122--1000_chkpt

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path "$CHKPT_DIR" \
--resum_vla_path "$CHKPT_DIR" \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False --num_images_in_input 2 --use_proprio True \
--use_lora True --use_fz False --use_minivlm True \
--image_aug True \
--batch_size 1 --grad_accumulation_steps 8 --learning_rate 2e-4 --lora_rank 64 \
--max_steps 5001 --save_freq 1000 --save_latest_checkpoint_only False \
--merge_lora_during_training True --use_pro_version True \
--libero_task_ids 0 \
--resume True --resume_step 1000 \
> logs/1task-1k--$current_time.log 2>&1 &
```

New checkpoints will be written under the same run ID (e.g. `--2000_chkpt`, `--3000_chkpt`, …). Keep other flags (dataset, LoRA, etc.) the same as the original run.

**3. Evaluate only task 0** on each checkpoint:
```bash
# Checkpoint is at outputs/<run_id>--1000_chkpt (run_id from training; append --1000_chkpt for 1k-step)
python -m experiments.robot.libero.run_libero_eval \
  --pretrained_checkpoint outputs/<your_run_id>--1000_chkpt \
  --task_suite_name libero_spatial \
  --task_ids 0 \
  --num_trials_per_task 20
```

**Parameters:**
- `--libero_task_ids 0` or `--libero_task_ids 0,1` — train on task(s) 0 and/or 1 only
- `--task_ids 0` — evaluate only task 0 (comma-separated for multiple)
- `--max_steps` — 1001, 5001, 10001 for 1k, 5k, 10k steps