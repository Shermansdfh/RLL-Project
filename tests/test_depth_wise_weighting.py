"""
Sanity checks for DepthWiseFeatureWeighting and its integration into the action head.

Run:
    python -m tests.test_depth_wise_weighting          (from repo root)
    python tests/test_depth_wise_weighting.py           (also works)

Checks:
    1. Shape / forward pass
    2. Backward pass / gradient flow
    3. Device transfer + parameter count
    4. Ultra-short smoke train (2 batches through L1RegressionActionHead)
"""

import sys
import os
import traceback

# ── make repo root importable when running as a script ──────────────────────
# We bypass prismatic/__init__.py (which pulls heavy deps like draccus)
# by loading the leaf .py files directly via importlib.util.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import importlib.util
import torch
import torch.nn as nn


def _load_module_from_file(name: str, filepath: str):
    """Load a single .py file as a module, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register so cross-imports within the file work
    spec.loader.exec_module(mod)
    return mod


# 1) constants must be loaded first (action_heads imports it)
_constants = _load_module_from_file(
    "prismatic.vla.constants",
    os.path.join(_REPO, "prismatic", "vla", "constants.py"),
)

# 2) action_heads
_action_heads = _load_module_from_file(
    "prismatic.models.action_heads",
    os.path.join(_REPO, "prismatic", "models", "action_heads.py"),
)

DepthWiseFeatureWeighting = _action_heads.DepthWiseFeatureWeighting
L1RegressionActionHead = _action_heads.L1RegressionActionHead
MLPResNet = _action_heads.MLPResNet

ACTION_DIM = _constants.ACTION_DIM
NUM_ACTIONS_CHUNK = _constants.NUM_ACTIONS_CHUNK
NUM_TOKENS = _constants.NUM_TOKENS
PROPRIO_DIM = _constants.PROPRIO_DIM

# ── reproducibility ─────────────────────────────────────────────────────────
torch.manual_seed(42)

# ── test dimensions ─────────────────────────────────────────────────────────
BATCH = 2
HIDDEN_DIM = 256          # smaller than prod (4096) to keep tests fast
NUM_VLM_LAYERS = 25       # embedding + 24 transformer layers
NUM_ACTION_HEAD_LAYERS = 24
NUM_PATCHES = 64          # stand-in for vision patches (prod ~512)
NUM_BLOCKS = 24

passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = ""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  [{tag}] {name}" + (f"  -- {detail}" if detail else ""))


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Shape / Forward Pass
# ═════════════════════════════════════════════════════════════════════════════
print("\n===== CHECK 1: Shape / Forward Pass =====")

# 1a. DepthWiseFeatureWeighting standalone
dw = DepthWiseFeatureWeighting(
    num_vlm_layers=NUM_VLM_LAYERS,
    num_action_head_layers=NUM_ACTION_HEAD_LAYERS,
    hidden_dim=HIDDEN_DIM,
    share_weights_across_layers=False,
    normalize_action_queries=True,
)
h_t_all = torch.randn(BATCH, NUM_VLM_LAYERS, NUM_PATCHES, HIDDEN_DIM)
h_a_all = torch.randn(BATCH, NUM_VLM_LAYERS, NUM_TOKENS, HIDDEN_DIM)

try:
    h_t_comb, h_a_comb = dw.precompute_all(h_t_all, h_a_all)
    # h_t_comb: (batch, N, num_patches, dim), h_a_comb: (batch, N, num_tokens, dim)
    h_t_out = h_t_comb[:, 0, :, :]
    h_a_out = h_a_comb[:, 0, :, :]
    ok_t = h_t_out.shape == (BATCH, NUM_PATCHES, HIDDEN_DIM)
    ok_a = h_a_out.shape == (BATCH, NUM_TOKENS, HIDDEN_DIM)
    report("DW forward runs", True)
    report(
        f"DW h_t shape: {tuple(h_t_out.shape)}",
        ok_t,
        f"expected ({BATCH}, {NUM_PATCHES}, {HIDDEN_DIM})",
    )
    report(
        f"DW h_a shape: {tuple(h_a_out.shape)}",
        ok_a,
        f"expected ({BATCH}, {NUM_TOKENS}, {HIDDEN_DIM})",
    )
except Exception as e:
    report("DW forward runs", False, str(e))
    traceback.print_exc()

# 1b. DepthWiseFeatureWeighting with shared weights
dw_shared = DepthWiseFeatureWeighting(
    num_vlm_layers=NUM_VLM_LAYERS,
    num_action_head_layers=NUM_ACTION_HEAD_LAYERS,
    hidden_dim=HIDDEN_DIM,
    share_weights_across_layers=True,
    normalize_action_queries=True,
)
try:
    h_t_sh, h_a_sh = dw_shared.precompute_all(h_t_all, h_a_all)
    # Shared weights: N=1, so all blocks get the same result
    ok = h_t_sh.shape[1] == 1 and h_a_sh.shape[1] == 1
    report("Shared-weights mode: single weight set (N=1)", ok)
except Exception as e:
    report("Shared-weights forward", False, str(e))

# 1c. normalize_action_queries=False path
dw_no_norm = DepthWiseFeatureWeighting(
    num_vlm_layers=NUM_VLM_LAYERS,
    num_action_head_layers=NUM_ACTION_HEAD_LAYERS,
    hidden_dim=HIDDEN_DIM,
    share_weights_across_layers=False,
    normalize_action_queries=False,
)
try:
    h_t_nn, h_a_nn = dw_no_norm.precompute_all(h_t_all, h_a_all)
    h_a_nn_0 = h_a_nn[:, 0, :, :]
    report("normalize_action_queries=False forward runs", True)
    report(
        f"h_a shape (no-norm): {tuple(h_a_nn_0.shape)}",
        h_a_nn_0.shape == (BATCH, NUM_TOKENS, HIDDEN_DIM),
    )
except Exception as e:
    report("normalize_action_queries=False forward", False, str(e))

# 1d. MLPResNet with depth-wise weighting
print()
net = MLPResNet(
    num_blocks=NUM_BLOCKS,
    input_dim=HIDDEN_DIM * ACTION_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=ACTION_DIM,
    use_pro_version=False,
    use_depth_wise_weighting=True,
    num_vlm_layers=NUM_VLM_LAYERS,
    share_depth_weights=False,
    normalize_aq_before_combination=True,
)

x_in = torch.randn(BATCH, NUM_ACTIONS_CHUNK, HIDDEN_DIM * ACTION_DIM)
h_t_in = torch.randn(BATCH, NUM_VLM_LAYERS, NUM_PATCHES, HIDDEN_DIM)
h_a_in = torch.randn(BATCH, NUM_VLM_LAYERS, NUM_TOKENS, HIDDEN_DIM)
p_in = torch.randn(BATCH, 1, HIDDEN_DIM)

try:
    out = net(x_in, h_a=h_a_in, h_t=h_t_in, p=p_in)
    expected_shape = (BATCH, NUM_ACTIONS_CHUNK, ACTION_DIM)
    ok = out.shape == expected_shape
    report(f"MLPResNet(DW) output shape: {tuple(out.shape)}", ok, f"expected {expected_shape}")
except Exception as e:
    report("MLPResNet(DW) forward", False, str(e))
    traceback.print_exc()

# 1e. MLPResNet WITHOUT depth-wise weighting (backward compat)
net_legacy = MLPResNet(
    num_blocks=NUM_BLOCKS,
    input_dim=HIDDEN_DIM * ACTION_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=ACTION_DIM,
    use_pro_version=False,
    use_depth_wise_weighting=False,
)
# Legacy path expects shape (batch, num_layers, seq, dim) and indexes [:, i+1, :]
# so dim-1 must be >= num_blocks+1 = 25
try:
    out_legacy = net_legacy(x_in, h_a=h_a_in, h_t=h_t_in, p=p_in)
    report(f"MLPResNet(legacy) output shape: {tuple(out_legacy.shape)}",
           out_legacy.shape == expected_shape)
except Exception as e:
    report("MLPResNet(legacy) forward", False, str(e))
    traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Backward Pass / Gradient Flow
# ═════════════════════════════════════════════════════════════════════════════
print("\n===== CHECK 2: Backward Pass / Gradient Flow =====")

net.zero_grad()
out = net(x_in, h_a=h_a_in, h_t=h_t_in, p=p_in)
loss = out.sum()
try:
    loss.backward()
    report("backward() completes without error", True)
except Exception as e:
    report("backward()", False, str(e))
    traceback.print_exc()

# Check gradients on key parameters
critical_params = {
    "feature_weighting.kv_weight_logits": net.feature_weighting.kv_weight_logits,
    "feature_weighting.aq_weight_logits": net.feature_weighting.aq_weight_logits,
    "feature_weighting.kv_layer_norms[0].weight": net.feature_weighting.kv_layer_norms[0].weight,
    "feature_weighting.aq_layer_norms[0].weight": net.feature_weighting.aq_layer_norms[0].weight,
    "fc1.weight": net.fc1.weight,
    "fc2.weight": net.fc2.weight,
    "mlp_resnet_blocks[0].q_proj.weight": net.mlp_resnet_blocks[0].q_proj.weight,
}

for name, p in critical_params.items():
    has_grad = p.grad is not None and p.grad.abs().sum() > 0
    report(f"grad flows to {name}", has_grad,
           "grad is None!" if p.grad is None else f"grad norm={p.grad.norm():.6f}")


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Device Transfer + Parameter Count
# ═════════════════════════════════════════════════════════════════════════════
print("\n===== CHECK 3: Device Transfer + Parameter Count =====")

total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"  MLPResNet(DW, dim={HIDDEN_DIM}) trainable params: {total_params:,}")
report("param count > 0", total_params > 0)
# Sanity: with dim=256 & 24 blocks, expect order of millions
report("param count in reasonable range (>100k)", total_params > 100_000,
       f"got {total_params:,}")

# CPU (always)
try:
    net_cpu = net.cpu()
    dummy_out = net_cpu(
        x_in.cpu(), h_a=h_a_in.cpu(), h_t=h_t_in.cpu(), p=p_in.cpu()
    )
    report("forward on CPU", True)
except Exception as e:
    report("forward on CPU", False, str(e))

# CUDA (if available)
if torch.cuda.is_available():
    try:
        dev = torch.device("cuda")
        net_cu = net.to(dev)
        dummy_out = net_cu(
            x_in.to(dev), h_a=h_a_in.to(dev), h_t=h_t_in.to(dev), p=p_in.to(dev)
        )
        report("forward on CUDA", True)
        # move back to CPU so later tests are device-agnostic
        net.cpu()
    except Exception as e:
        report("forward on CUDA", False, str(e))
else:
    print("  [SKIP] CUDA not available")

# MPS (if available on Mac)
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    try:
        dev = torch.device("mps")
        net_mps = net.to(dev)
        dummy_out = net_mps(
            x_in.to(dev), h_a=h_a_in.to(dev), h_t=h_t_in.to(dev), p=p_in.to(dev)
        )
        report("forward on MPS", True)
        net.cpu()
    except Exception as e:
        report("forward on MPS", False, str(e))
else:
    print("  [SKIP] MPS not available")


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Ultra-short Smoke Train (through L1RegressionActionHead)
# ═════════════════════════════════════════════════════════════════════════════
print("\n===== CHECK 4: Ultra-short Smoke Train =====")

# Build a small action head with depth-wise weighting enabled
action_head = L1RegressionActionHead(
    input_dim=HIDDEN_DIM,
    hidden_dim=HIDDEN_DIM,
    action_dim=ACTION_DIM,
    num_task_tokens=NUM_PATCHES,
    use_pro_version=False,
    use_depth_wise_weighting=True,
    num_vlm_layers=NUM_VLM_LAYERS,
    share_depth_weights=False,
    normalize_aq_before_combination=True,
).to(torch.bfloat16)

# Simple proprio projector (stand-in)
proprio_projector = nn.Linear(PROPRIO_DIM, HIDDEN_DIM).to(torch.bfloat16)

optimizer = torch.optim.AdamW(
    list(action_head.parameters()) + list(proprio_projector.parameters()),
    lr=1e-4,
)

NUM_BATCHES = 2
for step in range(NUM_BATCHES):
    optimizer.zero_grad()

    # Synthetic inputs matching the real pipeline shapes:
    #   actions_hidden_states: (batch, num_vlm_layers, NUM_PATCHES + NUM_TOKENS, hidden_dim)
    actions_hidden_states = torch.randn(
        BATCH, NUM_VLM_LAYERS, NUM_PATCHES + NUM_TOKENS, HIDDEN_DIM,
        dtype=torch.bfloat16,
    )
    proprio = torch.randn(BATCH, PROPRIO_DIM, dtype=torch.bfloat16)
    gt_actions = torch.randn(BATCH, NUM_ACTIONS_CHUNK, ACTION_DIM, dtype=torch.bfloat16)

    try:
        predicted = action_head.predict_action(
            actions_hidden_states,
            proprio=proprio,
            proprio_projector=proprio_projector,
            phase="Training",
        )
        loss = nn.L1Loss()(predicted, gt_actions)
        loss.backward()
        optimizer.step()
        print(f"  step {step}: loss = {loss.item():.6f}  shape = {tuple(predicted.shape)}")
        report(f"smoke train step {step}", True)
    except Exception as e:
        report(f"smoke train step {step}", False, str(e))
        traceback.print_exc()

# Also test with use_pro_version=True
print("\n  -- Pro version smoke --")
action_head_pro = L1RegressionActionHead(
    input_dim=HIDDEN_DIM,
    hidden_dim=HIDDEN_DIM,
    action_dim=ACTION_DIM,
    num_task_tokens=NUM_PATCHES,
    use_pro_version=True,
    use_depth_wise_weighting=True,
    num_vlm_layers=NUM_VLM_LAYERS,
    share_depth_weights=True,
    normalize_aq_before_combination=False,
).to(torch.bfloat16)

try:
    out_pro = action_head_pro.predict_action(
        actions_hidden_states,
        proprio=proprio,
        proprio_projector=proprio_projector,
        phase="Inference",
    )
    report(f"Pro version forward shape: {tuple(out_pro.shape)}",
           out_pro.shape == (BATCH, NUM_ACTIONS_CHUNK, ACTION_DIM))
except Exception as e:
    report("Pro version forward", False, str(e))
    traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  RESULTS:  {passed} passed,  {failed} failed")
print(f"{'='*60}")
sys.exit(1 if failed > 0 else 0)
