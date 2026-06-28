#!/usr/bin/env python3
"""
Generate a video of the trained TD3 agent running from a checkpoint.

Reads hyperparameters from each run's wandb config.yaml so that networks
with different hidden dimensions or environments load correctly.

Usage:
    python record_video.py                                           # latest-run
    python record_video.py -r rg8qmdij                               # run by short ID
    python record_video.py -r rg8qmdij --episodes 3 --device cuda
    python record_video.py --all --top 5                             # newest 5 runs
"""

import argparse
import os
import sys
import glob
import re

import yaml

# Prevent td3.py's module-level argparse / wandb.init from running on import.
_orig_argv = sys.argv[:]
sys.argv = ["td3.py", "--no_wandb"]
import td3  # noqa: E402 (import after sys.argv fix)
sys.argv = _orig_argv

WANDB_DIR = os.path.join(os.path.dirname(__file__), "wandb")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _resolve_run_dir(run_id):
    """Given a short run ID (e.g. 'rg8qmdij'), return the matching run dir path.

    Also accepts a full run-dir name like 'run-20260626_202149-rg8qmdij'
    or a full checkpoint path.
    """
    # Already a directory?
    if os.path.isdir(run_id):
        return run_id
    # Already a checkpoint path?  (strip trailing /td3_best.pt)
    if run_id.endswith("/td3_best.pt") or run_id.endswith("\\td3_best.pt"):
        d = os.path.dirname(run_id)
        if os.path.isdir(d):
            return d
    # Full run-dir name (e.g. run-20260626_202149-rg8qmdij)
    full = os.path.join(WANDB_DIR, run_id)
    if os.path.isdir(full):
        return full
    # Short ID — scan wandb/run-* for a match
    for d in sorted(glob.glob(os.path.join(WANDB_DIR, "run-*"))):
        if d.endswith("-" + run_id):
            return d
    raise FileNotFoundError(f"No run directory found for '{run_id}'")


def load_run_config(run_dir):
    """Parse files/config.yaml from a run directory and return a dict with
    the keys relevant for reconstruction.  Handles both 'hidden' (list) and
    'hidden1'/'hidden2' styles.
    """
    config_path = os.path.join(run_dir, "files", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # wandb config stores everything as {key: {value: ...}} at the top level.
    # Top-level keys that are dicts with 'value' are hyperparams.
    cfg = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "value" in v:
            cfg[k] = v["value"]

    # Normalise hidden dims: some runs use 'hidden', others 'hidden1'/'hidden2'
    hidden = cfg.get("hidden")
    if hidden is None:
        h1 = cfg.get("hidden1")
        h2 = cfg.get("hidden2")
        if h1 is not None and h2 is not None:
            hidden = [h1, h2]
    if hidden is None:
        hidden = [512, 256]  # fallback default from td3.py

    return {
        "env": cfg.get("env", "Ant-v4"),
        "hidden": tuple(hidden),
        "seed": cfg.get("seed", 42),
    }


def find_all_checkpoints(wandb_dir):
    """Return list of (run_dir, ckpt_path) tuples sorted by run dir name."""
    results = []
    for run_dir in sorted(glob.glob(os.path.join(wandb_dir, "run-*"))):
        ckpt = os.path.join(run_dir, "td3_best.pt")
        if os.path.exists(ckpt):
            results.append((run_dir, ckpt))
    # Also include latest-run if it exists
    latest_ckpt = os.path.join(wandb_dir, "latest-run", "td3_best.pt")
    latest_dir = os.path.join(wandb_dir, "latest-run")
    if os.path.exists(latest_ckpt) and latest_dir not in [r for r, _ in results]:
        results.append((latest_dir, latest_ckpt))
    return results


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------
def record_episode(ckpt_path, env_name, hidden, video_dir,
                   episode_idx=0, seed=42, device="cpu"):
    """Load a checkpoint and record one episode."""
    import numpy as np
    import torch
    import gymnasium as gym

    # Create env to get dimensions
    tmp_env = gym.make(env_name)
    state_dim = int(tmp_env.observation_space.shape[0])
    action_dim = int(tmp_env.action_space.shape[0])
    max_action = float(tmp_env.action_space.high[0])
    tmp_env.close()

    # Build agent with the correct architecture
    agent = td3.TD3(state_dim, action_dim, max_action,
                    tau=0.005, gamma=0.99,
                    policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                    hidden=hidden)
    agent.load(ckpt_path)
    agent.actor.to(device)
    agent.actor.eval()

    # Create video-recorded env
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda ep: True,
        name_prefix=f"td3-{os.path.basename(os.path.dirname(ckpt_path))}",
    )

    obs, _ = env.reset(seed=seed + episode_idx)
    total_reward = 0.0
    steps = 0

    while True:
        with torch.no_grad():
            state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action = agent.actor(state).squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if steps >= 300: truncated=True
        if terminated or truncated:
            break

    env.close()
    return total_reward, steps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Record video of trained TD3 agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-r", "--run", type=str, default=None,
                        help="Run ID (e.g. rg8qmdij) or run dir name. "
                             "Default: latest-run")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Explicit checkpoint path (overrides --run)")
    parser.add_argument("--env", type=str, default=None,
                        help="Override environment name (default: read from config)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu",
                        help="'cpu' or 'cuda'")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: ./videos)")
    parser.add_argument("--all", action="store_true",
                        help="Record from ALL checkpoints in wandb/")
    parser.add_argument("--top", type=int, default=0,
                        help="With --all: only record the newest N checkpoints")
    args = parser.parse_args()

    # Resolve output directory
    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), "videos")
    os.makedirs(args.output, exist_ok=True)

    # Build list of (run_dir, ckpt_path)
    if args.all:
        pairs = find_all_checkpoints(WANDB_DIR)
        if not pairs:
            print("No checkpoints found under", WANDB_DIR)
            sys.exit(1)
        if args.top > 0:
            pairs = pairs[-args.top:]
        print(f"Found {len(pairs)} checkpoints "
              f"(recording {'newest ' + str(args.top) if args.top else 'all'})")
    elif args.ckpt:
        # Explicit checkpoint path — treat its parent as the run dir
        ckpt = args.ckpt
        run_dir = os.path.dirname(ckpt)
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found: {ckpt}")
            sys.exit(1)
        pairs = [(run_dir, ckpt)]
    else:
        # --run (or default to latest-run)
        run_id = args.run if args.run else "latest-run"
        try:
            run_dir = _resolve_run_dir(run_id)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
        ckpt = os.path.join(run_dir, "td3_best.pt")
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found: {ckpt}")
            sys.exit(1)
        pairs = [(run_dir, ckpt)]

    for run_dir, ckpt_path in pairs:
        run_name = os.path.basename(run_dir)

        # Read hyperparams from this run's config
        try:
            cfg = load_run_config(run_dir)
        except FileNotFoundError as e:
            print(f"  Skipping {run_name}: {e}")
            continue

        env_name = args.env if args.env else cfg["env"]
        hidden = cfg["hidden"]

        print(f"\n{'='*60}")
        print(f"Run:           {run_name}")
        print(f"Checkpoint:    {ckpt_path}")
        print(f"Environment:   {env_name}")
        print(f"Hidden dims:   {list(hidden)}")
        print(f"{'='*60}")

        for ep in range(args.episodes):
            try:
                ret, steps = record_episode(
                    ckpt_path, env_name, hidden, args.output,
                    episode_idx=ep, seed=args.seed, device=args.device,
                )
                print(f"  Episode {ep+1}:  return={ret:.1f}  steps={steps}")
            except Exception as e:
                print(f"  Episode {ep+1}:  FAILED — {e}")

        print(f"  Video saved to: {args.output}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
