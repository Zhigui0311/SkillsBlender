from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG
import skillsblender.tasks.path.mdp as mdp


# ==============================================================================
# Scene 
# ==============================================================================
GO2_BODY_NAMES = ['base', 
                  'FL_hip', 'FL_thigh','FL_calf', 'FL_foot',
                  'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot', 
                  'Head_upper', 'Head_lower', 
                  'RL_hip','RL_thigh', 'RL_calf', 'RL_foot', 
                  'RR_hip','RR_thigh', 'RR_calf', 'RR_foot']

GO2_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
]

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
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
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshI
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
    },
)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene """

    # plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator", # "plane" , "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=1,
        collision_group=-1, 
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    #robots    
    robot: ArticulationCfg = MISSING

    #sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[3.0, 1.0]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        track_air_time=True
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=750.0, color=(0.75, 0.75, 0.75)),
    )



# ==============================================================================
# Events 
# ==============================================================================

@configclass
class EventCfg:
    """Configuration for environment events."""
    
    # startup events
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            # "recompute_inertia": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
        },
    )


    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

# ==============================================================================
# Curriculum Learning
# ==============================================================================
# 在 PathEnvCfg 中添加 curriculum 属性

# @configclass
# class CurriculumCfg:
#     """Curriculum learning configuration."""
#
#     # 路径长度课程: 从短路径逐步增加到长路径
#     path_length = CurrTerm(
#         func=mdp.curriculum_path_length,
#         params={
#             "command_name": "path_tracking",
#             "initial_range": (2.5, 3.5),  # 初始路径长度范围
#             "final_range": (2.5, 6.0),    # 最终路径长度范围
#             "reward_threshold": 50.0      # 平均奖励超过此值时增加难度
#         }
#     )
#
#     # 速度要求课程: 逐步提高速度要求 (可选)
#     # velocity_requirement = CurrTerm(
#     #     func=mdp.curriculum_velocity_requirement,
#     #     params={
#     #         "initial_velocity": 0.5,
#     #         "final_velocity": 1.5,
#     #         "reward_threshold": 60.0
#     #     }
#     # )

# ==============================================================================
# MDP Settings 
# ==============================================================================

@configclass
class CommandsCfg:
    """Commands specification for the MDP."""
    
    
    path_tracking = mdp.commands.PathCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 15.0), # resample every 10-15s
        
        inpoints=mdp.commands.PathCommandCfg.InterpolationPoints(
            path_type="linear",
            height_change=False,
            end_to_start_pos=(2.5, 6.0, 0), # 终点范围
            yaw_type="along_path",       
            start_heading=(-math.pi, 0), 
            end_heading=(0, math.pi),   
        ),
        
        ranges=mdp.commands.PathCommandCfg.Ranges(
            num_waypoints=100,           
            num_lookahead_waypoints=6,  
            waypoint_reach_threshold=0.8,
        ),
        debug_vis=True, 
    )


@configclass
class ActionsCfg:
    """Actions specification for the MDP."""
    joint_pos_actoion = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=GO2_JOINT_NAMES, scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Observations for the Policy Group."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Inputs to the policy network."""
        
        path_slice= ObsTerm(
            func=mdp.path_slice_obs, 
            scale = 1.0,
            params={"command_name": "path_tracking"}
        )
        
        # ---  Proprioception ---
        # base_lin_vel = ObsTerm(
        #     func=mdp.base_lin_vel, 
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-100.0,100.0),
        #     scale=1.0,
        # )
        
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0,100.0),
            scale=1.0,
        )
        
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise= Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0,100.0),
            scale=1.0,
        )
        
        # ---  Robot Joint States ---
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", preserve_order=True)},
            clip=(-100.0,100.0),
            scale=1.0,
        )
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", preserve_order=True)},
            clip=(-100.0,100.0),
            scale=1.0,
        )
        
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0,100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
        

    @configclass
    class CriticCfg(ObsGroup):
        """Observations to the critic network."""
        
        path_slice= ObsTerm(
            func=mdp.path_slice_obs, 
            params={"command_name": "path_tracking"}
        )
        
        # ---  (Proprioception) ---
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, 
            clip=(-100.0,100.0),
            scale=1.0,
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
            clip=(-100.0,100.0),
            scale=1.0,
        )
            
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0,100.0),
            scale=1.0,
        )
    
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", preserve_order=True)},
            clip=(-100.0,100.0),
            scale=1.0,
        )
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", preserve_order=True)},
            clip=(-100.0,100.0),
            scale=1.0,
        )

        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0,100.0),
            scale=1.0,
        )
        #可以考虑加上last last
        #height_scanner  加不加这个功能？现在不能加 暂时都不能有

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    policy = PolicyCfg()
    critic = CriticCfg()

@configclass
class RewardsCfg:
    """Reward function configuration."""

    # --- General ---
    is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)
    joint_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot")})
    
    # --- task ---
    track_xy = RewTerm(
        func=mdp.track_path_pos_xy_exp,
        weight=5.0,
        params={"std": 0.5, "command_name": "path_tracking"}
    )
    track_yaw = RewTerm(
        func=mdp.track_path_heading_exp,
        weight=2.0,
        params={"std": 0.5, "command_name": "path_tracking"}
    )

    stalling_penalty = RewTerm(
        func=mdp.stalling_penalty,
        weight=0,
        params={"command_name": "path_tracking" }
    )
    
    #base
    base_height_l2 = RewTerm(func=mdp.base_height_l2, params={"target_height": 0.34, "asset_cfg": SceneEntityCfg("robot")}, weight=-1.0)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot")})
    base_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    base_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    base_acc = RewTerm(func=mdp.base_acc, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot")})
    
    # Joint penalties
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES)}
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES)}
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES)}
    )

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES)}
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES), "soft_ratio": 1.0},
    )
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )
    
    # Action penalties
    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    
    # Contact sensor
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), #!!!
            "threshold": 1.0,
        },
    )
    
    undesired_contacts_hip = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_upper", "Head_lower", "RL_hip", "RR_hip"]),
            "threshold": 1.0
        },
    )

    # --- 基础运动质量奖励 ---
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        params={},
        weight=-2.0  # 惩罚机体倾斜
    )

    # # 关节加速度惩罚
    # joint_acc_l2 = RewTerm(
    #     func=mdp.joint_acc_l2,
    #     params={},
    #     weight=-2.5e-7
    # )

    # # 终点减速
    # near_goal_velocity = RewTerm(
    #     func=mdp.near_goal_velocity_penalty,
    #     weight=-2.0,
    #     params={
    #         "command_name": "path_tracking",
    #         "distance_threshold": 1.0,
    #         "max_velocity": 0.5
    #     }
    # )

    # # 终点稳定性
    # goal_stability = RewTerm(
    #     func=mdp.goal_reached_stability_reward,
    #     weight=3.0,
    #     params={
    #         "command_name": "path_tracking",
    #         "distance_threshold": 0.5,
    #         "velocity_threshold": 0.2
    #         }
    # )    
                                                                                                                                                     
    # # 添加hip关节角度约束                                                                                                                                                            
    # hip_angle_penalty = RewTerm(                                                         
    #     func=mdp.hip_joint_angle_penalty,                                                
    #     weight=-2.0,                                                                     
    #     params={                                                                         
    #         "asset_cfg": SceneEntityCfg("robot"),                                        
    #         "max_hip_angle": 0.15  # 约8.6度                                                                                                                                
    #       }                                                                                
    #    )

    # feet rewards
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_acc = RewTerm(
        func=mdp.feet_acceleration_penalty,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
    )
    
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    
    feet_gait = RewTerm(
        func=mdp.GaitReward,
        weight=0.0,
        params={
            "std": math.sqrt(0.5),
            "command_name": "path_tracking",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    
    feet_height = RewTerm(
        func=mdp.feet_height_body,
        weight=0.0,
        params={
            "command_name": "path_tracking",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": -0.2,
            "dis_threshold": 0.25,
            "heading_threshold": 0.5,
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_1,
        weight=0.0,
        params={
            "command_name": "path_tracking",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.5,
            "dis_threshold": 0.25,
            "heading_threshold": 0.5,
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )

@configclass
class TerminationsCfg:
    """Termination conditions for the MDP."""
    
    # 1. 倒地判定 (Base 接触地面)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), 
            "threshold": 1.0
        },
    )
    
    # 2. 超时
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 3. [Custom] 偏离路径太远终止
    path_deviation = DoneTerm(
        func=mdp.path_deviation,
        params={"min_threshold":1.0,"max_threshold": 10, "command_name": "path_tracking"},
    )

    # === 新增姿态终止 ===  termination
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.5},  # 约 30 度
        time_out=False
    )



# -- Environment Configuration 


@configclass
class PathEnvCfg(ManagerBasedRLEnvCfg):
    """
    Flat terrain environment configuration for robot navigating along a path.

    如需启用课程学习 (Curriculum Learning):
    1. 取消注释上面的 CurriculumCfg 类
    2. 在此类中添加: curriculum: CurriculumCfg = CurriculumCfg()
    """
    # scene
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0)

    # Observations, Actions, Commands
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # 如需启用课程学习，取消下面一行的注释 (需先取消 CurriculumCfg 的注释)
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        self.sim.dt = 0.005 # 200Hz Simulation frequency
        self.decimation = 4 # 50Hz control frequency
        self.episode_length_s = 10.0  # Episode length

        self.sim.render_interval = 2
        self.sim.physics_material = self.scene.terrain.physics_material
        self.viewer.asset_name = "robot"
        self.viewer.origin_type = "asset"
        self.viewer.eye = (3.0, 3.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)