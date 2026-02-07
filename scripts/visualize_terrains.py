#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain Visualization Script for SkillsBlender.

This script allows you to visualize different terrain types used in training
before starting the actual training process. It spawns the robot on each
terrain type and lets you explore the environment interactively.

Usage:
    # Visualize all available terrains
    python scripts/visualize_terrains.py

    # Visualize specific terrain type
    python scripts/visualize_terrains.py --terrain parkour_easy

    # Visualize with specific number of environments
    python scripts/visualize_terrains.py --terrain stairs --num_envs 16

    # List all available terrain types
    python scripts/visualize_terrains.py --list

Available Terrains:
    - flat: Flat terrain for basic walking
    - stairs: Pyramid stairs terrain
    - jump: Gap terrain for jumping
    - climb: Sloped terrain for climbing
    - crouch: Flat terrain with low obstacles for crouching
    - rough: Random rough terrain
    - boxes: Random grid boxes
    - parkour_easy: Easy parkour mix
    - parkour_medium: Medium parkour mix
    - parkour_hard: Hard parkour mix
    - blender: Multi-skill blender terrain
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Parse arguments before launching app
parser = argparse.ArgumentParser(description="Visualize different terrain types for training.")
parser.add_argument("--terrain", type=str, default="parkour_medium", help="Terrain type to visualize.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
parser.add_argument("--list", action="store_true", help="List all available terrain types.")
parser.add_argument("--spacing", type=float, default=5.0, help="Environment spacing.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# List terrains and exit if requested
if args_cli.list:
    print("\nAvailable Terrain Types:")
    print("=" * 50)
    terrains = [
        ("flat", "Flat plane terrain"),
        ("stairs", "Pyramid stairs (up/down)"),
        ("jump", "Gap terrain for jumping"),
        ("climb", "Sloped terrain for climbing"),
        ("crouch", "Flat with low obstacles for crouching"),
        ("rough", "Random rough terrain"),
        ("boxes", "Random grid boxes"),
        ("parkour_easy", "Easy parkour mix (30% flat)"),
        ("parkour_medium", "Medium parkour mix (20% flat)"),
        ("parkour_hard", "Hard parkour mix (10% flat)"),
        ("blender", "Multi-skill blender terrain"),
    ]
    for name, desc in terrains:
        print(f"  {name:20s} - {desc}")
    print()
    sys.exit(0)

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after app launch."""

import torch
import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG


# ==============================================================================
# Terrain Configurations
# ==============================================================================

FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)

STAIRS_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

JUMP_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=1.0,
            gap_width_range=(0.3, 0.9),
            platform_width=2.0,
        ),
    },
)

CLIMB_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=(0.10, 0.40),
            platform_width=2.0,
            border_width=0.25
        ),
    },
)

CROUCH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # 30% flat terrain
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        # 70% flat with low obstacles (for crouching under)
        # Note: Actual obstacles will be spawned separately in the scene
        "crouch_area": terrain_gen.MeshPlaneTerrainCfg(proportion=0.7),
    },
)

ROUGH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.02, 0.12),
            noise_step=0.02,
            border_width=0.25
        ),
    },
)

BOXES_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,
            grid_width=0.45,
            grid_height_range=(0.05, 0.25),
            platform_width=2.0
        ),
    },
)

# Parkour terrains (imported from parkour_env_cfg)
PARKOUR_EASY_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.5),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.30),
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.15,
            gap_width_range=(0.2, 0.5),
            platform_width=2.5,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.04, 0.10),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.05, 0.20),
            platform_width=2.5,
            border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.01, 0.05),
            noise_step=0.01,
            border_width=0.25
        ),
    },
)

PARKOUR_MEDIUM_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.3, 0.7),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.20,
            gap_width_range=(0.35, 0.70),
            platform_width=2.0,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.06, 0.15),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.10, 0.30),
            platform_width=2.0,
            border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.02, 0.08),
            noise_step=0.02,
            border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.40,
            grid_height_range=(0.05, 0.15),
            platform_width=2.0
        ),
    },
)

PARKOUR_HARD_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.5, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.20,
            gap_width_range=(0.5, 0.9),
            platform_width=1.8,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.08, 0.20),
            step_width=0.28,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.15, 0.40),
            platform_width=2.0,
            border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.03, 0.12),
            noise_step=0.02,
            border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.20,
            grid_width=0.45,
            grid_height_range=(0.08, 0.25),
            platform_width=2.0
        ),
    },
)

BLENDER_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.20,
            gap_width_range=(0.3, 0.8),
            platform_width=2.0,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.10, 0.40),
            platform_width=2.0,
            border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.05, 0.20),
            platform_width=2.0
        ),
    },
)

# Terrain registry
TERRAIN_REGISTRY = {
    "flat": FLAT_TERRAIN_CFG,
    "stairs": STAIRS_TERRAIN_CFG,
    "jump": JUMP_TERRAIN_CFG,
    "climb": CLIMB_TERRAIN_CFG,
    "crouch": CROUCH_TERRAIN_CFG,
    "rough": ROUGH_TERRAIN_CFG,
    "boxes": BOXES_TERRAIN_CFG,
    "parkour_easy": PARKOUR_EASY_TERRAIN_CFG,
    "parkour_medium": PARKOUR_MEDIUM_TERRAIN_CFG,
    "parkour_hard": PARKOUR_HARD_TERRAIN_CFG,
    "blender": BLENDER_TERRAIN_CFG,
}


# ==============================================================================
# Scene Configuration
# ==============================================================================

@configclass
class TerrainVisualizationSceneCfg(InteractiveSceneCfg):
    """Scene configuration for terrain visualization."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FLAT_TERRAIN_CFG,  # Will be overridden
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.9, 0.9, 1.0)),
    )


@configclass
class CrouchVisualizationSceneCfg(TerrainVisualizationSceneCfg):
    """Scene configuration for crouch terrain with low obstacles."""

    # Main low roof: long and wide, so the robot must crouch instead of bypassing it.
    crouch_roof = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CrouchRoof",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.8, 0.0, 0.34),  # center height; bottom ~0.29 for 0.10 thickness
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(2.6, 1.6, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.3)),
        ),
    )

    # Tail roof extends low-clearance section so policy must hold crouch for longer.
    crouch_roof_tail = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CrouchRoofTail",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.0, 0.0, 0.34),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 1.4, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.3)),
        ),
    )


def main():
    """Main function to visualize terrains."""
    # Get terrain configuration
    terrain_name = args_cli.terrain.lower()
    if terrain_name not in TERRAIN_REGISTRY:
        print(f"[ERROR] Unknown terrain type: {terrain_name}")
        print(f"Available terrains: {list(TERRAIN_REGISTRY.keys())}")
        return

    terrain_cfg = TERRAIN_REGISTRY[terrain_name]
    print(f"\n[INFO] Visualizing terrain: {terrain_name}")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    print(f"[INFO] Environment spacing: {args_cli.spacing}")

    # Print terrain composition
    print(f"\n[INFO] Terrain composition:")
    for name, sub_cfg in terrain_cfg.sub_terrains.items():
        proportion = sub_cfg.proportion * 100
        print(f"  - {name}: {proportion:.0f}%")

    # Create scene configuration (use special config for crouch terrain)
    if terrain_name == "crouch":
        scene_cfg = CrouchVisualizationSceneCfg(
            num_envs=args_cli.num_envs,
            env_spacing=args_cli.spacing,
        )
        print(f"\n[INFO] Added crouch obstacles (low roofs) to scene")
    else:
        scene_cfg = TerrainVisualizationSceneCfg(
            num_envs=args_cli.num_envs,
            env_spacing=args_cli.spacing,
        )
    scene_cfg.terrain.terrain_generator = terrain_cfg

    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device if args_cli.device else "cuda:0",
        dt=0.005,
        render_interval=2,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[10.0, 10.0, 10.0], target=[0.0, 0.0, 0.0])

    # Create scene
    scene = InteractiveScene(scene_cfg)

    # Reset scene
    sim.reset()
    scene.reset()

    print("\n" + "=" * 60)
    print("TERRAIN VISUALIZATION")
    print("=" * 60)
    print(f"Terrain: {terrain_name}")
    print(f"Environments: {args_cli.num_envs}")
    print()
    print("Controls:")
    print("  - Use mouse to rotate camera")
    print("  - Scroll to zoom in/out")
    print("  - Press 'R' to reset camera")
    print("  - Press 'ESC' or close window to exit")
    print("=" * 60 + "\n")

    # Simulation loop
    while simulation_app.is_running():
        # Step simulation
        sim.step()

        # Update scene
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Visualization stopped by user.")
    finally:
        simulation_app.close()
