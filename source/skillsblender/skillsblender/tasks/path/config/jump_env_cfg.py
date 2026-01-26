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
from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg
import skillsblender.tasks.path.mdp as mdp


# ==============================================================================
# Scene 
# ==============================================================================
JUMP_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    num_rows=16,
    num_cols=24,
    curriculum=False,
    difficulty_range=(0.0,1.0),
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    border_width = 0.5,
    sub_terrains={
        # 1. 基础平地
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.0), # 禁用平地，全部使用gap地形
        
        # 2. 窄沟壑
        "narrow_gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.5,
            gap_width_range=(0.3, 0.5),               
            platform_width=2.0,           
        ),
        
        # 3. 宽沟壑：用于进阶跳跃训练 (占比 40%)
        "wide_gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.5,
            gap_width_range=(0.6, 1.0),  # 沟壑宽度 0.6m - 1.0m (挑战 Go2 极限)
            platform_width=2.5,
        ),
    },
)


GO2_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
]

@configclass
class MyJumpSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene """

    # plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator", 
        terrain_generator=JUMP_TERRAIN_CFG,
        max_init_terrain_level=None,
        collision_group=-1, 
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
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
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[6.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

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
# MDP Settings 
# ==============================================================================

@configclass
class CommandsCfg:       
    """Commands specification for the MDP."""
    
    
    path_tracking = mdp.commands.JumpPathCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 15.0), # resample every 10-15s
        
        jump_params=mdp.commands.JumpPathCommandCfg.JumpParams(
            jump_height=0.35,           # Max height of the parabolic arc
            gap_threshold = -0.4 ,         # Height drop to identify a gap (meters)
            scan_dist= 6.0,          # How far to look ahead for gaps
            scan_step = 0.1,            # Resolution of terrain scanning
            takeoff_margin= 0.2,       # Distance before gap to start arc
            landing_margin = 0.3, 
        ),
        ranges=mdp.commands.JumpPathCommandCfg.Ranges(
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
        asset_name="robot", 
        joint_names=GO2_JOINT_NAMES,  
        # joint_names=[".*"], 
        scale=0.25, 
        use_default_offset=True, 
        clip={".*": (-100.0, 100.0)}
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
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES , preserve_order=True)},
            clip=(-100.0,100.0),
            scale=1.0,
        )
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES, preserve_order=True)},
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
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES, preserve_order=True)},
            clip=(-100.0,100.0),
            scale=1.0,
        )
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES, preserve_order=True)},
            clip=(-100.0,100.0),
            scale=1.0,
        )

        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0,100.0),
            scale=1.0,
        )
        #可以考虑加上last last
        #height_scanner  加不加这个功能？

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    policy = PolicyCfg()
    critic = CriticCfg()

@configclass
class RewardsCfg:
    """Reward function configuration for jump training."""

    # --- Task rewards (基础路径追踪) ---
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
    track_velocity = RewTerm(
        func=mdp.track_velocity_along_path_exp,
        weight=3.0,
        params={"std": 0.6, "command_name": "path_tracking"}
    )

    # --- Jump-specific rewards (跳跃专用奖励) ---
    # 跳跃高度追踪：鼓励机器人跟随抛物线轨迹
    jump_height_tracking = RewTerm(
        func=mdp.jump_height_tracking,
        weight=3.0,
        params={
            "command_name": "path_tracking",
            "height_tolerance": 0.15
        }
    )

    # 跳跃前向速度：鼓励保持足够的前向动力
    jump_forward_velocity = RewTerm(
        func=mdp.jump_forward_velocity,
        weight=2.0,
        params={
            "target_velocity": 1.5,
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    # 跳跃离地高度：鼓励跳得足够高，避免碰到沟壑边缘
    jump_clearance = RewTerm(
        func=mdp.jump_clearance_reward,
        weight=1.5,
        params={
            "min_clearance": 0.2,
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "path_tracking"
        }
    )

    # 跳跃落地稳定性：鼓励稳定落地
    jump_landing_stability = RewTerm(
        func=mdp.jump_landing_stability,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "landing_time_threshold": 0.2,
            "velocity_threshold": 0.5
        }
    )

    # 跳跃俯仰角控制：鼓励合理的俯仰角
    jump_pitch_control = RewTerm(
        func=mdp.jump_pitch_control,
        weight=1.0,
        params={
            "target_pitch_range": (-0.2, 0.2),
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    # 跳跃腾空时间：奖励合理的腾空时间
    jump_air_time = RewTerm(
        func=mdp.jump_air_time_reward,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "min_air_time": 0.3,
            "max_air_time": 1.5
        }
    )

    # --- Regularization (正则化 - 防止动作乱动) ---
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-0.0001)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # 关节姿态正则化：鼓励保持默认站姿
    joint_dev = RewTerm(func=mdp.joint_deviation_l2, weight=-0.1)

    # 非脚部碰撞惩罚
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*thigh"),
            "threshold": 1.0
        },
    )

    # 存活奖励
    alive_rew = RewTerm(func=mdp.is_alive, weight=1.0)



@configclass
class TerminationsCfg:
    """Termination conditions for jump training."""

    # 基础终止条件
    # 倒地判定
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0
        },
    )

    # 超时
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 偏离路径
    path_deviation = DoneTerm(
        func=mdp.path_deviation,
        params={"min_threshold": 1.0, "max_threshold": 10, "command_name": "path_tracking"},
    )

    # --- Jump-specific terminations (跳跃专用终止条件) ---
    # 掉入沟壑
    jump_gap_fall = DoneTerm(
        func=mdp.jump_gap_fall,
        params={
            "height_threshold": -0.5,
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    # 过度旋转（翻滚）
    jump_excessive_rotation = DoneTerm(
        func=mdp.jump_excessive_rotation,
        params={
            "max_roll": 1.2,  # 约70度
            "max_pitch": 1.2,  # 约70度
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    # 落地失败
    jump_landing_failure = DoneTerm(
        func=mdp.jump_landing_failure,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "min_contact_feet": 2,
            "check_after_air_time": 0.5,
            "velocity_threshold": 2.0
        }
    )

    # 长时间无进展
    jump_no_progress = DoneTerm(
        func=mdp.jump_timeout_no_progress,
        params={
            "command_name": "path_tracking",
            "time_threshold": 5.0,
            "min_progress": 0.1
        }
    )

    # 卡在沟壑里
    jump_stuck = DoneTerm(
        func=mdp.jump_stuck_in_gap,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "time_threshold": 2.0,
            "velocity_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

# ==================================================== 帮我看看go2_jump_cfg.py还需要改吗
# @configclass
# class JumpCurriculumCfg:
#     """Jump-specific curriculum learning configuration."""
#
#     # 沟壑宽度课程: 从窄沟壑逐步增加到宽沟壑
#     jump_gap_width = CurrTerm(
#         func=mdp.curriculum_jump_gap_width,
#         params={
#             "reward_threshold": 50.0,           # 平均奖励超过此值时增加难度
#             "initial_gap_range": (0.3, 0.5),   # 初始沟壑宽度范围
#             "final_gap_range": (0.6, 1.0),     # 最终沟壑宽度范围
#             "step_size": 0.1                    # 每次增加步长
#         }
#     )
#
#     # 跳跃高度要求课程: 逐步提高跳跃高度
#     jump_height_requirement = CurrTerm(
#         func=mdp.curriculum_jump_height_requirement,
#         params={
#             "command_name": "path_tracking",
#             "reward_threshold": 60.0,
#             "initial_height": 0.25,            # 初始跳跃高度
#             "final_height": 0.45,              # 最终跳跃高度
#             "step_size": 0.05
#         }
#     )
#
#     # 地形混合课程: 逐步增加跳跃地形比例
#     jump_terrain_mix = CurrTerm(
#         func=mdp.curriculum_jump_terrain_mix,
#         params={
#             "reward_threshold": 55.0,
#             "max_jump_proportion": 0.9         # 跳跃地形的最大比例
#         }
#     )
#
#     # 速度要求课程: 逐步提高跳跃时的速度要求 (可选)
#     # jump_speed_requirement = CurrTerm(
#     #     func=mdp.curriculum_jump_speed_requirement,
#     #     params={
#     #         "reward_threshold": 65.0,
#     #         "initial_speed": 1.0,
#     #         "final_speed": 2.0,
#     #         "step_size": 0.1
#     #     }
#     # )




# -- Environment Configuration 


@configclass
class JumpPathEnvCfg(ManagerBasedRLEnvCfg):
    """
    Jump terrain environment configuration for robot learning to jump over gaps.

    如需启用跳跃课程学习 (Jump Curriculum Learning):
    1. 取消注释上面的 JumpCurriculumCfg 类
    2. 在此类中添加: curriculum: JumpCurriculumCfg = JumpCurriculumCfg()
    """
    # scene
    scene: MyJumpSceneCfg = MyJumpSceneCfg(num_envs=4096, env_spacing=10.0)

    # Observations, Actions, Commands
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # 如需启用跳跃课程学习，取消下面一行的注释 (需先取消 JumpCurriculumCfg 的注释)
    # curriculum: JumpCurriculumCfg = JumpCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        self.sim.dt = 0.005 # 200Hz Simulation frequency
        self.decimation = 4 # 50Hz control frequency
        self.episode_length_s = 15.0  # 跳跃任务需要更长的episode时间

        self.sim.render_interval = 2
        self.sim.physics_material = self.scene.terrain.physics_material
        self.viewer.asset_name = "robot"
        self.viewer.origin_type = "asset"
        self.viewer.eye = (3.0, 3.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
