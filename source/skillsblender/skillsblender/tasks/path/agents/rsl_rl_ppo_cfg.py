# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
from skillsblender.tasks.path.agents.symmetry import GO2

@configclass
class PathRslRlPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 5000
    save_interval = 100
    experiment_name = "path_rsl_rl_ppo"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=10,
        num_mini_batches=32,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )
    
    
@configclass
class GO2JumpPPOWithSymmetryCfg(PathRslRlPPOCfg):
    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 200
    experiment_name = "go2-path-jump"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=16, # Increased for augmented data
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
        use_data_augmentation=True,
        data_augmentation_func=GO2.compute_symmetric_states#这个函数需要的env obs参数怎么传递进来的？
        )
    )


# ==============================================================================
# New Skill PPO Configurations
# ==============================================================================

@configclass
class GO2WalkPPOCfg(PathRslRlPPOCfg):
    """Go2 walk skill PPO configuration."""
    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 200
    experiment_name = "go2-path-walk"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=GO2.compute_symmetric_states
        )
    )


@configclass
class GO2WalkSlopePPOCfg(GO2WalkPPOCfg):
    """Go2 walk+slope PPO configuration."""
    experiment_name = "go2-path-walk-slope"


@configclass
class GO2StairsPPOCfg(PathRslRlPPOCfg):
    """Go2 stairs skill PPO configuration."""
    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 200
    experiment_name = "go2-path-stairs"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=GO2.compute_symmetric_states
        )
    )


@configclass
class GO2PlatformPPOCfg(PathRslRlPPOCfg):
    """Go2 platform climb skill PPO configuration."""
    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 200
    experiment_name = "go2-path-platform"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=GO2.compute_symmetric_states
        )
    )


@configclass
class GO2CrouchPPOCfg(PathRslRlPPOCfg):
    """Go2 crouch skill PPO configuration."""
    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 200
    experiment_name = "go2-path-crouch"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=GO2.compute_symmetric_states
        )
    )


@configclass
class GO2BlenerPPOCfg(PathRslRlPPOCfg):
    """Go2 multi-skill blender PPO configuration."""
    num_steps_per_env = 64  # Longer rollouts for complex terrain
    max_iterations = 5000  # More iterations for multi-skill learning
    save_interval = 200
    experiment_name = "go2-path-blener"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=GO2.compute_symmetric_states
        )
    )


@configclass
class GO2VirtualSkillPPOCfg(PathRslRlPPOCfg):
    """Base PPO config for virtual-hallucination skills."""
    num_steps_per_env = 64
    max_iterations = 4000
    save_interval = 200
    experiment_name = "go2-path-virtual-skill"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=GO2.compute_symmetric_states,
        ),
    )


@configclass
class GO2VirtualJumpPPOCfg(GO2VirtualSkillPPOCfg):
    experiment_name = "go2-path-virtual-jump"


@configclass
class GO2VirtualCrouchPPOCfg(GO2VirtualSkillPPOCfg):
    experiment_name = "go2-path-virtual-crouch"

@configclass
class GO2ClimbPPOCfg(PathRslRlPPOCfg):
    """Go2 climb skill PPO configuration."""
    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 200
    experiment_name = "go2-path-climb"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=GO2.compute_symmetric_states
        )
    )
