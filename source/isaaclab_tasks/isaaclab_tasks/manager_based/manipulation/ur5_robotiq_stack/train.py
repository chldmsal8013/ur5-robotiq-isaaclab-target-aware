"""
train.py — UR5 + Robotiq 2F-85 Cube Stacking (vision PPO)

Usage:
    ~/IsaacLab/isaaclab.sh -p train.py --task Isaac-Stack-UR5-Robotiq-v0 --num_envs 512

Background (KITSU02):
    nohup ~/IsaacLab/isaaclab.sh -p train.py \
        --task Isaac-Stack-UR5-Robotiq-v0 \
        --num_envs 1024 \
        > training_log1.txt 2>&1 &
"""

import sys
import os

# ── Path injection (required for custom task discovery on KITSU02) ────────────
# Adjust this path to wherever you placed the ur5_robotiq_stack folder
TASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

# ── Isaac Sim MUST be imported first ─────────────────────────────────────────
import isaacsim  # noqa: F401

# ── Isaac Lab imports ─────────────────────────────────────────────────────────
from isaaclab.app import AppLauncher

# CLI
import argparse
parser = argparse.ArgumentParser(description="Train UR5 stacking PPO")
parser.add_argument("--task",       type=str,   default="Isaac-Stack-UR5-Robotiq-v0")
parser.add_argument("--num_envs",   type=int,   default=512)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--max_iterations", type=int, default=30000)
parser.add_argument("--checkpoint", type=str,   default=None, help="Resume from checkpoint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Headless by default for server training. CLI --headless still works, but the
# task should not silently launch GUI mode when the AppLauncher arg exists.
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Task registration ─────────────────────────────────────────────────────────

import gymnasium as gym
import os
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401


def main():
    # ── Make env ─────────────────────────────────────────────────────────────
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.curriculum = None  # DISABLED

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # ── Runner ───────────────────────────────────────────────────────────────
    from ur5_robotiq_stack.agents.rsl_rl_ppo_cfg import UR5RobotiqStackPPORunnerCfg
    agent_cfg = UR5RobotiqStackPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations

    log_root = os.path.join(os.path.expanduser("~"), "IsaacLab", "logs", "rsl_rl")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_root, device="cuda:0")

    # Resume from checkpoint if provided
    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)
        print(f"[train.py] Resumed from: {args_cli.checkpoint}")

    # ── Train ────────────────────────────────────────────────────────────────
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
