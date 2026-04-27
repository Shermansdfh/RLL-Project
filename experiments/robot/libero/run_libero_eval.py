"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.libero.private_object_split import (
    get_private_libero_object_split,
    get_private_libero_object_task_infos,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class LayerwiseCosineSimilarityCollector:
    """Aggregate per-query layerwise cosine-similarity summaries during eval."""

    output_dir: Path
    run_id: str
    enabled: bool = False
    matrix_sum: Optional[np.ndarray] = None
    num_queries: int = 0
    num_layers: int = 0
    num_tokens: int = 0
    token_slice: str = "all"

    def update(self, diagnostics: Optional[dict]) -> None:
        if not self.enabled or diagnostics is None:
            return

        matrix = np.asarray(diagnostics["cosine_similarity"], dtype=np.float64)
        if self.matrix_sum is None:
            self.matrix_sum = np.zeros_like(matrix)
            self.num_layers = int(diagnostics["num_layers"])
            self.num_tokens = int(diagnostics["num_tokens"])
            self.token_slice = str(diagnostics["token_slice"])

        self.matrix_sum += matrix
        self.num_queries += 1

    def finalize(self, log_file=None) -> None:
        if not self.enabled or self.matrix_sum is None or self.num_queries == 0:
            log_message("Layerwise cosine similarity collection skipped or produced no samples.", log_file)
            return

        mean_matrix = (self.matrix_sum / float(self.num_queries)).astype(np.float32)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        matrix_path = self.output_dir / "vlm_layer_cosine_similarity.npy"
        metadata_path = self.output_dir / "vlm_layer_cosine_similarity.json"
        figure_path = self.output_dir / "vlm_layer_cosine_similarity.png"

        np.save(matrix_path, mean_matrix)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": self.run_id,
                    "num_queries": self.num_queries,
                    "num_layers": self.num_layers,
                    "num_tokens_per_layer": self.num_tokens,
                    "token_slice": self.token_slice,
                    "mean_diagonal": float(np.diag(mean_matrix).mean()),
                    "mean_off_diagonal": float(
                        (mean_matrix.sum() - np.trace(mean_matrix))
                        / max(mean_matrix.size - len(mean_matrix), 1)
                    ),
                },
                f,
                indent=2,
            )

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mean_matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_title(f"VLM layer cosine similarity ({self.token_slice} tokens)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(self.num_layers))
        ax.set_yticks(range(self.num_layers))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")
        fig.tight_layout()
        fig.savefig(figure_path, dpi=200)
        plt.close(fig)

        log_message(
            f"Saved layerwise cosine similarity matrix to {matrix_path} and heatmap to {figure_path}",
            log_file,
        )


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_minivlm: bool = True                         # If True, uses minivlm
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    libero_object_private_split: Optional[str] = None  # Shared Stage 1 / Stage 2 LIBERO-Object split preset
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    save_version: str = "vla-adapter"                # version of 
    use_pro_version: bool = True                     # encourage to use the pro models we released.
    phase: str = "Inference"
    dry_run: bool = False                            # If True, resolve split/config and exit before model init

    # Depth-wise feature weighting (must match training flags)
    use_depth_wise_weighting: bool = False            # If True, uses learnable layer-wise mixing
    share_depth_weights: bool = False                 # If True, all action-head layers share mixing weights
    normalize_aq_before_combination: bool = True      # If True, LayerNorm ActionQueries before combining
    depth_weight_top_k: int = 0                       # If >0, each action-head block keeps only top-k VLM layers
    depth_weight_epsilon: float = 0.0                 # Epsilon-greedy: prob. of replacing top-k with a random k (train-only)
    depth_weight_activation: str = "softmax"           # Activation for mixing logits: "softmax", "sparsemax", or "entmax15"
    use_action_queries: bool = True                      # If False, VLM skips learnable action-query injection
    use_kv_gate: bool = True                             # If False, action-head blocks drop the tanh(g) gate on VLM-KV attention
    eval_vlm_layer_cosine_similarity: bool = False       # If True, save cosine similarity across VLM layers during eval
    eval_vlm_layer_cosine_token_slice: str = "all"       # Which hidden states to compare: "all", "task", or "action"



def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.eval_vlm_layer_cosine_token_slice in {"all", "task", "action"}, (
        "eval_vlm_layer_cosine_token_slice must be one of: all, task, action"
    )

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    if cfg.libero_object_private_split is not None:
        split = get_private_libero_object_split(cfg.libero_object_private_split)
        assert cfg.task_suite_name == split.task_suite_name, (
            f"`--libero_object_private_split {cfg.libero_object_private_split}` requires "
            f"`--task_suite_name {split.task_suite_name}`."
        )



def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    # Pure-KV-cache ablation: disable learnable action-query injection into the VLM
    model.use_action_queries = cfg.use_action_queries
    model.eval_layer_cosine_enabled = cfg.eval_vlm_layer_cosine_similarity
    model.eval_layer_cosine_token_slice = cfg.eval_vlm_layer_cosine_token_slice
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key



def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    artifact_dir = os.path.join(cfg.local_log_dir, run_id)
    os.makedirs(artifact_dir, exist_ok=True)
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id, artifact_dir



def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def print_bottleneck_diagnosis(timing_dict: dict, log_file=None, prefix: str = ""):
    """Print mean timing (ms) and % of step for each phase. timing_dict has list values in seconds."""
    if not timing_dict or all(len(v) == 0 for v in timing_dict.values()):
        return
    step_phases = ["prepare_obs", "env_step"]
    policy_phases = ["preprocess_transfer", "model_forward"]
    step_data = [(k, timing_dict[k]) for k in step_phases if timing_dict.get(k)]
    policy_data = [(k, timing_dict[k]) for k in policy_phases if timing_dict.get(k)]
    lines = [f"{prefix}Bottleneck diagnosis:"]
    if step_data:
        total_s = sum(sum(v) / len(v) if v else 0.0 for _, v in step_data)
        n = len(step_data[0][1])
        lines.append(f"  Per env step (n_steps={n}):")
        for k, v in step_data:
            mean_ms = (sum(v) / len(v)) * 1000 if v else 0
            pct = 100.0 * (sum(v) / len(v)) / total_s if total_s else 0
            lines.append(f"    {k:<22} {mean_ms:>10.2f} ms  {pct:>5.1f}%")
        lines.append(f"    {'total':<22} {total_s*1000:>10.2f} ms  100.0%")
    if policy_data:
        total_s = sum(sum(v) / len(v) if v else 0.0 for _, v in policy_data)
        n = len(policy_data[0][1])
        lines.append(f"  Per policy query (n_queries={n}):")
        for k, v in policy_data:
            mean_ms = (sum(v) / len(v)) * 1000 if v else 0
            pct = 100.0 * (sum(v) / len(v)) / total_s if total_s else 0
            lines.append(f"    {k:<22} {mean_ms:>10.2f} ms  {pct:>5.1f}%")
        lines.append(f"    {'total':<22} {total_s*1000:>10.2f} ms  100.0%")
    log_message("\n".join(lines), log_file)


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None



def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay



def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action



def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    timing_dict=None,
    layerwise_cosine_collector: Optional[LayerwiseCosineSimilarityCollector] = None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
               "{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            t_prep = time.perf_counter()
            observation, img = prepare_observation(obs, resize_size)
            if timing_dict is not None:
                timing_dict.setdefault("prepare_obs", []).append(time.perf_counter() - t_prep)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm,
                    timing_dict=timing_dict,
                )
                if layerwise_cosine_collector is not None:
                    layerwise_cosine_collector.update(getattr(model, "_last_hidden_state_diagnostics", None))

                action_queue.extend(actions) 

            # Get action from queue
            action = action_queue.popleft()
            # action = actions[0]


            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            t_env = time.perf_counter()
            obs, reward, done, info = env.step(action.tolist())
            if timing_dict is not None:
                timing_dict.setdefault("env_step", []).append(time.perf_counter() - t_env)
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images




def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    save_version=None,
    layerwise_cosine_collector: Optional[LayerwiseCosineSimilarityCollector] = None,
):
    """Run evaluation for a single task."""
    # Get task
    # task_id = 8
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Timing for bottleneck diagnosis
    timing_dict = {"prepare_obs": [], "preprocess_transfer": [], "model_forward": [], "env_step": []}

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            timing_dict=timing_dict,
            layerwise_cosine_collector=layerwise_cosine_collector,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file, save_version=save_version
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    print_bottleneck_diagnosis(timing_dict, log_file=log_file, prefix=f"[Task {task_id}] ")

    # close env
    env.close()
    del env

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes



@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    selected_task_infos = None
    selected_task_ids = None
    if cfg.libero_object_private_split is not None:
        selected_task_infos = get_private_libero_object_task_infos(cfg.libero_object_private_split)
        selected_task_ids = [task.task_id for task in selected_task_infos]
        logger.info(
            "Using LIBERO-Object private split `%s`: task_ids=%s",
            cfg.libero_object_private_split,
            selected_task_ids,
        )
        if cfg.dry_run:
            for task in selected_task_infos:
                logger.info("Selected task %d: %s | %s", task.task_id, task.task_name, task.language)
            return 0.0

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # for name, param in model.named_parameters():
    #     if 'action_queries' in name: 
    #         print(f"{name}: {param}")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id, artifact_dir = setup_logging(cfg)
    layerwise_cosine_collector = LayerwiseCosineSimilarityCollector(
        output_dir=Path(artifact_dir),
        run_id=run_id,
        enabled=cfg.eval_vlm_layer_cosine_similarity,
    )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    if selected_task_ids is None:
        selected_task_ids = list(range(task_suite.n_tasks))

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Selected task ids: {selected_task_ids}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(selected_task_ids):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            cfg.save_version,
            layerwise_cosine_collector=layerwise_cosine_collector,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    layerwise_cosine_collector.finalize(log_file=log_file)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate



if __name__ == "__main__":
    eval_libero()
