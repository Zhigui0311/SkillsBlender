#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train multi-skill parkour policy with knowledge distillation.

This script trains a unified student policy by distilling knowledge from
pre-trained expert policies for each skill (walk, jump, stairs, climb, platform, crouch).

The distillation is implemented as an additional loss term that encourages
the student policy to match the expert actions for each skill phase.

Usage:
    # Train with distillation from expert policies
    python scripts/rsl_rl/train_distill.py --task go2-parkour-distill-v0 \\
        --walk_expert "logs/rsl_rl/go2-path-walk/*/model_*.pt" \\
        --jump_expert "logs/rsl_rl/go2-path-jump/*/model_*.pt" \\
        --stairs_expert "logs/rsl_rl/go2-path-stairs/*/model_*.pt" \\
        --climb_expert "logs/rsl_rl/go2-path-climb/*/model_*.pt" \\
        --platform_expert "logs/rsl_rl/go2-path-platform/*/model_*.pt" \\
        --crouch_expert "logs/rsl_rl/go2-path-crouch/*/model_*.pt" \\
        --distill_coef 1.0

    # Train without experts (falls back to standard PPO)
    python scripts/rsl_rl/train_distill.py --task go2-parkour-distill-v0
"""

import argparse
import sys
import glob

from isaaclab.app import AppLauncher

import cli_args

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train multi-skill policy with distillation.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="go2-parkour-distill-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# Expert policy paths (support glob patterns)
parser.add_argument("--walk_expert", type=str, default=None, help="Path to walk expert checkpoint.")
parser.add_argument("--jump_expert", type=str, default=None, help="Path to jump expert checkpoint.")
parser.add_argument("--stairs_expert", type=str, default=None, help="Path to stairs expert checkpoint.")
parser.add_argument("--climb_expert", type=str, default=None, help="Path to climb expert checkpoint.")
parser.add_argument("--platform_expert", type=str, default=None, help="Path to platform expert checkpoint.")
parser.add_argument("--crouch_expert", type=str, default=None, help="Path to crouch expert checkpoint.")

# Distillation parameters
parser.add_argument("--distill_coef", type=float, default=1.0, help="Distillation loss coefficient.")

# Append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after app launch."""

import gymnasium as gym
import logging
import os
import torch
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, Optional, List

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import skillsblender.tasks  # noqa: F401


logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def resolve_checkpoint_path(pattern: Optional[str]) -> Optional[str]:
    """Resolve glob pattern to actual checkpoint path."""
    if pattern is None:
        return None
    matches = glob.glob(pattern)
    if not matches:
        logger.warning(f"No checkpoint found matching: {pattern}")
        return None
    # Return the most recent checkpoint
    return max(matches, key=os.path.getmtime)


class ExpertEnsemble:
    """Ensemble of expert policies for different skills."""

    SKILL_NAMES = ["walk", "jump", "stairs_up", "stairs_down", "climb", "crouch"]

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        device: str,
        expert_paths: Dict[str, Optional[str]],
        hidden_dims: List[int] = None,
    ):
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.device = device
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.hidden_dims = hidden_dims
        self.experts: Dict[str, Optional[ActorCritic]] = {}

        # Load expert policies
        for skill_name in self.SKILL_NAMES:
            path = expert_paths.get(skill_name)
            if path and os.path.exists(path):
                try:
                    self.experts[skill_name] = self._load_expert(path)
                    print(f"[INFO] Loaded {skill_name} expert from: {path}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {skill_name} expert: {e}")
                    self.experts[skill_name] = None
            else:
                self.experts[skill_name] = None

        # Use walk expert as fallback
        self.fallback_expert = self.experts.get("walk")

    def _load_expert(self, path: str) -> ActorCritic:
        """Load expert policy from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Create actor-critic with same architecture
        expert = ActorCritic(
            num_actor_obs=self.num_obs,
            num_critic_obs=self.num_obs,
            num_actions=self.num_actions,
            actor_hidden_dims=self.hidden_dims,
            critic_hidden_dims=self.hidden_dims,
            activation="elu",
        ).to(self.device)

        # Load weights - handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            expert.load_state_dict(checkpoint["model_state_dict"])
        else:
            expert.load_state_dict(checkpoint)

        expert.eval()
        for param in expert.parameters():
            param.requires_grad = False

        return expert

    @torch.no_grad()
    def get_expert_actions(
        self,
        observations: torch.Tensor,
        skill_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Get actions from appropriate expert based on skill phase."""
        N = observations.shape[0]
        expert_actions = torch.zeros(N, self.num_actions, device=self.device)

        for skill_idx, skill_name in enumerate(self.SKILL_NAMES):
            mask = skill_ids == skill_idx
            if not mask.any():
                continue

            expert = self.experts.get(skill_name) or self.fallback_expert
            if expert is None:
                continue

            obs_subset = observations[mask]
            expert_actions[mask] = expert.act_inference(obs_subset)

        return expert_actions

    def has_any_expert(self) -> bool:
        """Check if any expert is loaded."""
        return any(e is not None for e in self.experts.values())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with distillation from expert policies."""
    # Override configurations
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Setup logging
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env_cfg.log_dir = log_dir

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap environment
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Resolve expert checkpoint paths
    expert_paths = {
        "walk": resolve_checkpoint_path(args_cli.walk_expert),
        "jump": resolve_checkpoint_path(args_cli.jump_expert),
        "stairs_up": resolve_checkpoint_path(args_cli.stairs_expert),
        "stairs_down": resolve_checkpoint_path(args_cli.stairs_expert),
        "climb": resolve_checkpoint_path(args_cli.climb_expert),
        "platform": resolve_checkpoint_path(args_cli.platform_expert),
        "crouch": resolve_checkpoint_path(args_cli.crouch_expert),
    }

    print("[INFO] Expert policies:")
    for skill, path in expert_paths.items():
        status = path if path else "Not loaded"
        print(f"  {skill}: {status}")

    # Get dimensions
    num_obs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    # Create expert ensemble
    expert_ensemble = ExpertEnsemble(
        num_obs=num_obs,
        num_actions=num_actions,
        device=agent_cfg.device,
        expert_paths=expert_paths,
    )

    use_distillation = expert_ensemble.has_any_expert() and args_cli.distill_coef > 0
    if not use_distillation:
        print("[INFO] No experts loaded or distill_coef=0. Training with standard PPO.")
    else:
        print(f"[INFO] Training with distillation (coef={args_cli.distill_coef})")

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # Load checkpoint if resuming
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # Save configuration
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Store distillation config for the training loop
    if use_distillation:
        # Monkey-patch the algorithm's update to include distillation
        original_update = runner.alg.update

        def update_with_distillation():
            """Wrapper that adds distillation loss to PPO update."""
            # Get current observations and skill IDs
            obs = runner.obs
            unwrapped_env = env.unwrapped

            # Get skill IDs from command manager
            skill_ids = torch.zeros(env.num_envs, dtype=torch.long, device=agent_cfg.device)
            if hasattr(unwrapped_env, "command_manager"):
                try:
                    cmd = unwrapped_env.command_manager.get_term("path_tracking")
                    if hasattr(cmd, "current_skill_id"):
                        skill_ids = cmd.current_skill_id
                except Exception:
                    pass

            # Get expert actions
            expert_actions = expert_ensemble.get_expert_actions(obs, skill_ids)

            # Store for use in loss computation
            runner.alg._expert_actions = expert_actions
            runner.alg._distill_coef = args_cli.distill_coef

            # Call original update
            return original_update()

        runner.alg.update = update_with_distillation

    # Run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
