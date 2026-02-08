#!/usr/bin/env python3
"""Path Generation Visualization Script for SkillsBlender.

This script visualizes the generated paths for different skills and terrains
without running the actual training. It helps you verify that the path generation
logic is working as expected.

Usage:
    # Visualize walk path on flat terrain
    python scripts/visualize_paths.py --task go2-path-walk-v0

    # Visualize jump path
    python scripts/visualize_paths.py --task go2-path-jump-v0

    # Visualize parkour paths
    python scripts/visualize_paths.py --task go2-parkour-direct-v0

    # Visualize with more environments
    python scripts/visualize_paths.py --task go2-path-stairs-v0 --num_envs 16

    # Regenerate paths every N seconds
    python scripts/visualize_paths.py --task go2-path-crouch-v0 --regenerate_interval 5.0
"""

import argparse
import torch

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Visualize path generation for different skills.")
parser.add_argument("--task", type=str, default="go2-path-walk-v0", help="Task/environment to visualize")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--regenerate_interval", type=float, default=10.0, help="Regenerate paths every N seconds (0 to disable)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

# Append AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time

import isaaclab_tasks  # noqa: F401
import skillsblender.tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv


def main():
    """Visualize path generation for the specified task."""

    # Create environment
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 5.0

    # Enable debug visualization
    if hasattr(env_cfg, "commands"):
        if hasattr(env_cfg.commands, "path_tracking"):
            env_cfg.commands.path_tracking.debug_vis = True

    # Disable some events for cleaner visualization
    if hasattr(env_cfg, "events"):
        env_cfg.events.push_robot = None
        env_cfg.events.base_external_force_torque = None

    # Set viewer to better position
    env_cfg.viewer.eye = (5.0, 5.0, 3.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    print("\n" + "="*80)
    print(f"Path Visualization for: {args_cli.task}")
    print("="*80)
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Regenerate interval: {args_cli.regenerate_interval}s")
    print("\nVisualization Legend:")
    print("  - Green small spheres: Path waypoints")
    print("  - Yellow sphere: Path start point")
    print("  - Red sphere: Path goal point")
    print("  - Blue arrow: Robot's actual heading")
    print("  - Green arrow: Desired heading from path")
    print("\nControls:")
    print("  - Press 'R' to manually regenerate paths")
    print("  - Press 'ESC' to exit")
    print("="*80 + "\n")

    # Reset environment to generate initial paths
    obs, _ = env.reset(seed=args_cli.seed)

    # Get command manager to access path data
    command_manager = env.command_manager
    path_command = None
    for term_name, term in command_manager._terms.items():
        if hasattr(term, "pos_path_w"):
            path_command = term
            break

    if path_command is None:
        print("[ERROR] Could not find path command term in the environment!")
        print("Make sure the task has a path tracking command.")
        env.close()
        return

    print(f"[INFO] Found path command: {type(path_command).__name__}")
    print(f"[INFO] Number of waypoints per path: {path_command.num_waypoints}")

    # Print path statistics
    print_path_statistics(path_command, env)

    # Simulation loop
    last_regenerate_time = time.time()
    step_count = 0

    while simulation_app.is_running():
        # Check if we should regenerate paths
        current_time = time.time()
        if args_cli.regenerate_interval > 0 and (current_time - last_regenerate_time) >= args_cli.regenerate_interval:
            print(f"\n[INFO] Regenerating paths at step {step_count}...")
            env.reset()
            print_path_statistics(path_command, env)
            last_regenerate_time = current_time

        # Take a zero action (robot stays in place)
        action = torch.zeros(env.action_space.shape, device=env.device)
        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1

        # Print info every 100 steps
        if step_count % 100 == 0:
            print(f"[INFO] Step {step_count} - Paths are being visualized...")

    # Close environment
    env.close()


def print_path_statistics(path_command, env):
    """Print statistics about the generated paths."""
    print("\n" + "-"*80)
    print("Path Statistics:")
    print("-"*80)

    # Get path data
    pos_path = path_command.pos_path_w  # (N, W, 3)
    heading_path = path_command.heading_path_w  # (N, W, 1)

    N = pos_path.shape[0]
    W = pos_path.shape[1]

    # Calculate path lengths
    path_lengths = torch.zeros(N, device=pos_path.device)
    for i in range(W - 1):
        segment_length = torch.norm(pos_path[:, i+1, :2] - pos_path[:, i, :2], dim=-1)
        path_lengths += segment_length

    print(f"  Number of paths: {N}")
    print(f"  Waypoints per path: {W}")
    print(f"  Path length (mean): {path_lengths.mean().item():.2f}m")
    print(f"  Path length (min): {path_lengths.min().item():.2f}m")
    print(f"  Path length (max): {path_lengths.max().item():.2f}m")

    # Height statistics
    z_min = pos_path[:, :, 2].min(dim=1).values
    z_max = pos_path[:, :, 2].max(dim=1).values
    z_range = z_max - z_min

    print(f"  Height range (mean): {z_range.mean().item():.3f}m")
    print(f"  Height range (max): {z_range.max().item():.3f}m")

    # Heading statistics
    heading_change = torch.abs(torch.diff(heading_path[:, :, 0], dim=1))
    total_heading_change = heading_change.sum(dim=1)

    print(f"  Total heading change (mean): {torch.rad2deg(total_heading_change.mean()).item():.1f}°")
    print(f"  Max single heading change: {torch.rad2deg(heading_change.max()).item():.1f}°")

    # Segment information (if available)
    if hasattr(path_command, "_seg_skill_id"):
        seg_skill_id = path_command._seg_skill_id
        seg_s0 = path_command._seg_s0
        seg_s1 = path_command._seg_s1

        # Count segments per environment
        valid_segs = (seg_skill_id >= 0).sum(dim=1)
        print(f"\n  Segments per path (mean): {valid_segs.float().mean().item():.1f}")

        # Skill distribution
        if hasattr(path_command, "SKILL_NAMES"):
            skill_names = path_command.SKILL_NAMES
            print(f"\n  Skill distribution:")
            for skill_id, skill_name in enumerate(skill_names):
                count = (seg_skill_id == skill_id).sum().item()
                if count > 0:
                    print(f"    {skill_name}: {count} segments")

    print("-"*80 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
