# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##


gym.register(
    id="go2-path-flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_cfg:Go2PathEnvCfg",
        "rsl_rl_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="go2-path-flat-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_cfg:Go2PathPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)