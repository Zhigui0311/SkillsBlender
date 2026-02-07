"""Crouch skill environment configuration."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg, MySceneCfg
import skillsblender.tasks.path.mdp as mdp


# ==============================================================================
# Crouch Terrain Configuration
# ==============================================================================

CROUCH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # 30% flat terrain
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        # 70% flat with low obstacles (for crouching under)
        # Note: Actual obstacles will be spawned separately
        "crouch_area": terrain_gen.MeshPlaneTerrainCfg(proportion=0.7),
    },
)


@configclass
class MyCrouchSceneCfg(MySceneCfg):
    """Crouch scene configuration with low obstacles."""

    # Main low roof: short obstacle (0.4m) that robot must crouch under.
    crouch_roof = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CrouchRoof",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.15, 0.0, 0.34),  # center height; bottom ~0.29 for 0.10 thickness
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 1.6, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # Tail roof removed - using single 0.4m obstacle instead of extended section
    # crouch_roof_tail = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/CrouchRoofTail",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(3.0, 0.0, 0.34),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #     ),
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.2, 1.4, 0.10),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=True,
    #             disable_gravity=True,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    # )

    def __post_init__(self):
        super().__post_init__()
        # Update terrain for crouch task
        self.terrain.terrain_generator = CROUCH_TERRAIN_CFG


# ==============================================================================
# Crouch Environment Configuration
# ==============================================================================

@configclass
class CrouchPathEnvCfg(PathEnvCfg):
    """Crouch skill training environment configuration."""

    def __post_init__(self):
        super().__post_init__()

        # Use crouch scene
        self.scene: MyCrouchSceneCfg = MyCrouchSceneCfg(num_envs=4096, env_spacing=2.5)
        self.commands.path_tracking.class_type = mdp.commands.CrouchPathCommand
        self.commands.path_tracking.sampling.end_to_start_pos = (3.0, 5.0, 0.0)
        self.commands.path_tracking.sampling.sample_goal_distance = True

        # Crouch-specific command configuration
        self.commands.path_tracking.ranges.num_waypoints = 80
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 24
        self.commands.path_tracking.ranges.default_path_len = 3.0
        self.commands.path_tracking.crouch_params.start_dist_range = (0.8, 1.2)
        self.commands.path_tracking.crouch_params.crouch_len = 0.4
        self.commands.path_tracking.crouch_params.base_height_ref = 0.24

        # Keep starts aligned with the roof corridor so every episode practices crouching.
        self.events.reset_base.params["pose_range"]["x"] = (-0.2, 0.2)
        self.events.reset_base.params["pose_range"]["y"] = (-0.15, 0.15)
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.15, 0.15)
        self.events.push_robot = None

        # Crouch-specific rewards
        self.rewards.track_xy.weight = 5.0
        self.rewards.track_yaw.weight = 2.0
        self.rewards.track_velocity_along_path_exp.weight = 2.0
        self.rewards.track_velocity_along_path_exp.params["desired_speed"] = 0.7  # Slower crawl speed

        # Low height reward (encourage crouching)
        self.rewards.base_height_l2.weight = -0.5
        self.rewards.base_height_l2.params["target_height"] = 0.24
        self.rewards.crouch_base_height_phase.weight = -3.0
        self.rewards.crouch_base_height_phase.params["target_height"] = 0.24

        # Stability in low stance
        self.rewards.flat_orientation.weight = -1.5

        # Avoid hitting obstacles with body
        self.rewards.undesired_contacts.weight = -8.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            "base", "Head_upper", "Head_lower", ".*_hip", ".*_thigh"
        ]
        self.rewards.undesired_contacts_hip.weight = -3.0

        # Any head/base collision with roof is a hard failure.
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base", "Head_upper", "Head_lower"
        ]
        self.terminations.base_contact.params["threshold"] = 0.6

        # Camera: show both robot and obstacle corridor during crouch.
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "robot"
        self.viewer.body_name = "base"
        # Third-person follow cam: above/behind, looking slightly ahead.
        self.viewer.eye = (-2.0, 1.0, 1.1)
        self.viewer.lookat = (1.5, 0.0, 0.25)

        # Smooth motion
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_acc_l2.weight = -2.5e-7
