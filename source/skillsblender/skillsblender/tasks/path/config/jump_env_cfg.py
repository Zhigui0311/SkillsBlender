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
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    curriculum= True,
    difficulty_range=(0.0,1.0),
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    border_width = 0.5,
    sub_terrains={
        # 1. 基础平地
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1), #但是我的训练的逻辑是规划路径然后训练
        
        # 2. 窄沟壑
        "narrow_gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.6,
            ditch_width_range=(0.3, 0.5),  
            ditch_depth=0.6,              
            platform_width=2.0,           
        ),
        
        # 3. 宽沟壑：用于进阶跳跃训练 (占比 40%)
        "wide_gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.3,
            ditch_width_range=(0.6, 1.0),  # 沟壑宽度 0.6m - 1.0m (挑战 Go2 极限)
            ditch_depth=1.0,
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
        max_init_terrain_level=1,
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
class CommandsCfg:       #这里要大改
    """Commands specification for the MDP."""
    
    
    path_tracking = mdp.commands.PathCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 15.0), # resample every 10-15s
        
        inpoints=mdp.commands.PathCommandCfg.InterpolationPoints(
            path_type="linear",
            height_change=False,
            end_to_start_pos=(2.0, 10.0,0), # 终点范围
            yaw_type="decoupled",       
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
    """Reward function configuration."""
    
    # --- task (基于 path_tracking 指令生成的误差指标) ---
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
    # track_z = RewTerm(
    #     func=mdp.track_path_height_exp, 
    #     weight=0.5, 
    #     params={"std": 0.1, "command_name": "path_tracking"}
    # )

    # --- normalization and penalties ---(防止动作乱动、提升平滑度)
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
    """Termination conditions for the MDP."""
    
    #  倒地判定 
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
        params={"min_threshold":1.0,"max_threshold": 10, "command_name": "path_tracking"},
    )



# -- Environment Configuration 


@configclass
class PathEnvCfg(ManagerBasedRLEnvCfg):
    """
    Flat terrain environment configuration for robot navigating along a path.
    """
    # scene
    scene: InteractiveSceneCfg = MyJumpSceneCfg(num_envs=4096, env_spacing=10.0)
    
    # Observations, Actions, Commands
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # MDP
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        
        self.sim.dt = 0.005 # 200Hz Simulation frequency
        self.decimation = 4 # 50Hz control frequency
        self.episode_length_s = 10.0 

        self.sim.render_interval = 2  
        # self.sim.physics_material = self.scene.terrain.physics_material
        # self.viewer.eye = (3.0, 3.0, 3.0)
        # self.viewer.lookat = (0.0, 0.0, 0.0)