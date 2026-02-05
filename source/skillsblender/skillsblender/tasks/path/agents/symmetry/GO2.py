# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Symmetry for GO2 path tracking task (policy/critic obs + actions).

This implementation applies left-right symmetry with two guarantees:
1) Only the first ``4 * num_lookahead_waypoints`` entries of command/path observation are mirrored.
2) Joint/action symmetry uses an explicit GO2 joint order:
   [FR, FL, RR, RL] x [hip, thigh, calf].
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

# Tail dimensions after command/path term in policy/critic observations.
# policy tail: base_ang_vel(3) + projected_gravity(3) + joint_pos(12) + joint_vel(12) + actions(12)
_POLICY_TAIL_DIM = 42
# critic tail: base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + joint_pos(12) + joint_vel(12) + actions(12)
_CRITIC_TAIL_DIM = 45

# Explicit GO2 joint order expected by symmetry mapping.
# This matches GO2_JOINT_NAMES in path env configs and action term ordering.
_CANONICAL_GO2_ORDER = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

# Left-right swap permutation under canonical order above.
# FR <-> FL and RR <-> RL.
_LR_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
# Hip joints (after swapping) require sign flip.
_HIP_SIGN_FLIP_IDXS = [0, 3, 6, 9]


def _get_num_lookahead_waypoints(env, command_name: str) -> int:
    """Read lookahead waypoint count from command term."""
    env_unwrapped = getattr(env, "unwrapped", env)
    cmd = env_unwrapped.command_manager.get_term(command_name)
    if hasattr(cmd, "num_lookahead_waypoints"):
        return int(cmd.num_lookahead_waypoints)
    return int(cmd.cfg.ranges.num_lookahead_waypoints)


@torch.no_grad()
def compute_symmetric_states(
    env,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
    command_name: str = "path_tracking",
):
    """Augment observations/actions using left-right symmetry.

    For command/path observations, only the first ``4 * W`` entries are mirrored,
    where ``W = num_lookahead_waypoints``. Command meta is preserved.
    """
    path_slice_dim = 0
    if obs is not None:
        if env is None:
            raise RuntimeError("`env` is required for symmetry when obs is provided.")
        w = _get_num_lookahead_waypoints(env, command_name)
        path_slice_dim = 4 * w
    else:
        w = 0

    # ---------------- observations ----------------
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)

        if "policy" in obs.keys():
            policy = obs["policy"]
            cmd_dim = int(policy.shape[-1]) - _POLICY_TAIL_DIM
            if cmd_dim < path_slice_dim:
                raise RuntimeError(
                    f"policy cmd dim ({cmd_dim}) is smaller than mirrored path dim ({path_slice_dim})."
                )
            obs_aug["policy"][:batch_size] = policy
            obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(
                policy,
                cmd_dim=cmd_dim,
                path_slice_dim=path_slice_dim,
                w=w,
            )

        if "critic" in obs.keys():
            critic = obs["critic"]
            cmd_dim = int(critic.shape[-1]) - _CRITIC_TAIL_DIM
            if cmd_dim < path_slice_dim:
                raise RuntimeError(
                    f"critic cmd dim ({cmd_dim}) is smaller than mirrored path dim ({path_slice_dim})."
                )
            obs_aug["critic"][:batch_size] = critic
            obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(
                critic,
                cmd_dim=cmd_dim,
                path_slice_dim=path_slice_dim,
                w=w,
            )
    else:
        obs_aug = None

    # ---------------- actions ----------------
    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


# =============================================================================
# Obs transforms (LEFT-RIGHT)
# =============================================================================


def _transform_policy_obs_left_right(
    obs: torch.Tensor,
    *,
    cmd_dim: int,
    path_slice_dim: int,
    w: int,
) -> torch.Tensor:
    """Policy obs layout:
    [command(path+meta)] + [base_ang_vel] + [projected_gravity] + [joint_pos] + [joint_vel] + [actions]
    """
    obs = obs.clone()
    device = obs.device

    s_path = slice(0, path_slice_dim)
    s_ang = slice(cmd_dim, cmd_dim + 3)
    s_grav = slice(s_ang.stop, s_ang.stop + 3)
    s_jpos = slice(s_grav.stop, s_grav.stop + 12)
    s_jvel = slice(s_jpos.stop, s_jpos.stop + 12)
    s_act = slice(s_jvel.stop, s_jvel.stop + 12)

    # Mirror only xyz-yaw slices, keep command meta untouched.
    obs[:, s_path] = _transform_path_slice_left_right(obs[:, s_path], w=w)

    # IsaacLab left-right convention for base terms.
    obs[:, s_ang] *= torch.tensor([-1.0, 1.0, -1.0], device=device)
    obs[:, s_grav] *= torch.tensor([1.0, -1.0, 1.0], device=device)

    obs[:, s_jpos] = _switch_go2_joints_left_right(obs[:, s_jpos])
    obs[:, s_jvel] = _switch_go2_joints_left_right(obs[:, s_jvel])
    obs[:, s_act] = _switch_go2_joints_left_right(obs[:, s_act])

    return obs


def _transform_critic_obs_left_right(
    obs: torch.Tensor,
    *,
    cmd_dim: int,
    path_slice_dim: int,
    w: int,
) -> torch.Tensor:
    """Critic obs layout:
    [command(path+meta)] + [base_lin_vel] + [base_ang_vel] + [projected_gravity] + [joint_pos] + [joint_vel] + [actions]
    """
    obs = obs.clone()
    device = obs.device

    s_path = slice(0, path_slice_dim)
    s_lin = slice(cmd_dim, cmd_dim + 3)
    s_ang = slice(s_lin.stop, s_lin.stop + 3)
    s_grav = slice(s_ang.stop, s_ang.stop + 3)
    s_jpos = slice(s_grav.stop, s_grav.stop + 12)
    s_jvel = slice(s_jpos.stop, s_jpos.stop + 12)
    s_act = slice(s_jvel.stop, s_jvel.stop + 12)

    # Mirror only xyz-yaw slices, keep command meta untouched.
    obs[:, s_path] = _transform_path_slice_left_right(obs[:, s_path], w=w)

    obs[:, s_lin] *= torch.tensor([1.0, -1.0, 1.0], device=device)
    obs[:, s_ang] *= torch.tensor([-1.0, 1.0, -1.0], device=device)
    obs[:, s_grav] *= torch.tensor([1.0, -1.0, 1.0], device=device)

    obs[:, s_jpos] = _switch_go2_joints_left_right(obs[:, s_jpos])
    obs[:, s_jvel] = _switch_go2_joints_left_right(obs[:, s_jvel])
    obs[:, s_act] = _switch_go2_joints_left_right(obs[:, s_act])

    return obs


def _transform_path_slice_left_right(path_flat: torch.Tensor, *, w: int) -> torch.Tensor:
    """Mirror path slice in body frame for [x, y, z, yaw_rel] x W.

    Left-right mirror: y -> -y, yaw_rel -> -yaw_rel.
    """
    if path_flat.numel() == 0:
        return path_flat
    b = path_flat.shape[0]
    path = path_flat.view(b, w, 4).clone()
    path[..., 1] *= -1.0
    path[..., 3] *= -1.0
    return path.view(b, 4 * w)


# =============================================================================
# Action / joint transforms
# =============================================================================


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    actions[:] = _switch_go2_joints_left_right(actions[:])
    return actions


def _switch_go2_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap left/right legs and flip hip signs with explicit GO2 order.

    Expected order is exactly `_CANONICAL_GO2_ORDER`.
    """
    if joint_data.shape[-1] != len(_CANONICAL_GO2_ORDER):
        raise RuntimeError(
            f"GO2 symmetry expects {len(_CANONICAL_GO2_ORDER)} joints, got {joint_data.shape[-1]}."
        )

    joint_data_switched = joint_data[..., _LR_PERM].clone()
    joint_data_switched[..., _HIP_SIGN_FLIP_IDXS] *= -1.0
    return joint_data_switched
