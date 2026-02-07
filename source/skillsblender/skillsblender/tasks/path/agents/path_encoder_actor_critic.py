"""1D-CNN Path Encoder for multi-skill path following.

This module provides a CNN-based path encoder that processes path slice observations
(batch, K, 4) -> (batch, feature_dim) to capture spatial patterns in the path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rsl_rl.modules import ActorCritic


class PathEncoderCNN(nn.Module):
    """1D-CNN module for encoding path slice observations.

    Architecture:
        Conv1d(4, 32, k=3, p=1) -> ELU -> Conv1d(32, 64, k=3, p=1) -> ELU
        -> AdaptiveAvgPool1d(1) -> Linear(64, feature_dim)

    Input shape: (batch, 4, K) where K is num_lookahead_waypoints
    Output shape: (batch, feature_dim)
    """

    def __init__(
        self,
        num_channels: int = 4,
        hidden_channels: tuple[int, int] = (32, 64),
        feature_dim: int = 32,
        activation: str = "elu",
    ):
        super().__init__()

        self.num_channels = num_channels
        self.feature_dim = feature_dim

        # Select activation function
        if activation.lower() == "elu":
            act_fn = nn.ELU
        elif activation.lower() == "relu":
            act_fn = nn.ReLU
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh
        else:
            act_fn = nn.ELU

        # Build CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(num_channels, hidden_channels[0], kernel_size=3, padding=1),
            act_fn(),
            nn.Conv1d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1),
            act_fn(),
            nn.AdaptiveAvgPool1d(1),  # (batch, hidden_channels[1], 1)
        )

        self.fc = nn.Linear(hidden_channels[1], feature_dim)

    def forward(self, path_slice: torch.Tensor) -> torch.Tensor:
        """Encode path slice observations.

        Args:
            path_slice: Path slice tensor of shape (batch, K, 4) or (batch, 4, K)
                        where K is the number of lookahead waypoints.

        Returns:
            Encoded features of shape (batch, feature_dim)
        """
        # Ensure input is (batch, channels, seq_len) for Conv1d
        if path_slice.dim() == 2:
            # Assume flattened input (batch, K*4), reshape to (batch, K, 4)
            batch_size = path_slice.shape[0]
            total_features = path_slice.shape[1]
            K = total_features // self.num_channels
            path_slice = path_slice.view(batch_size, K, self.num_channels)

        if path_slice.shape[-1] == self.num_channels:
            # Input is (batch, K, 4), transpose to (batch, 4, K)
            path_slice = path_slice.transpose(1, 2)

        # Apply CNN
        x = self.conv_layers(path_slice)  # (batch, hidden_channels[1], 1)
        x = x.squeeze(-1)  # (batch, hidden_channels[1])
        x = self.fc(x)  # (batch, feature_dim)

        return x


class ActorCriticWithPathEncoder(nn.Module):
    """Actor-Critic network with CNN path encoder.

    This module wraps the standard RSL-RL ActorCritic and adds a CNN encoder
    for path slice observations. The encoded path features are concatenated
    with other observations before being passed to the MLP.

    The observation is expected to have path slice features at the beginning,
    followed by other proprioceptive observations.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        # Path encoder parameters
        path_slice_dim: int = 96,  # K * 4, e.g., 24 * 4 = 96
        path_encoder_hidden: tuple[int, int] = (32, 64),
        path_encoder_feature_dim: int = 32,
        # MLP parameters (passed to base ActorCritic)
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.path_slice_dim = path_slice_dim
        self.num_channels = 4  # x_fwd, y_lat, z_rel, dyaw

        # Create path encoder
        self.path_encoder = PathEncoderCNN(
            num_channels=self.num_channels,
            hidden_channels=path_encoder_hidden,
            feature_dim=path_encoder_feature_dim,
            activation=activation,
        )

        # Calculate new observation dimensions after path encoding
        # Original: path_slice (path_slice_dim) + other_obs
        # New: encoded_path (path_encoder_feature_dim) + other_obs
        other_actor_obs_dim = num_actor_obs - path_slice_dim
        other_critic_obs_dim = num_critic_obs - path_slice_dim

        new_actor_obs_dim = path_encoder_feature_dim + other_actor_obs_dim
        new_critic_obs_dim = path_encoder_feature_dim + other_critic_obs_dim

        # Select activation function
        if activation.lower() == "elu":
            act_fn = nn.ELU
        elif activation.lower() == "relu":
            act_fn = nn.ReLU
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh
        else:
            act_fn = nn.ELU

        # Build actor MLP
        actor_layers = []
        prev_dim = new_actor_obs_dim
        for hidden_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(act_fn())
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Build critic MLP
        critic_layers = []
        prev_dim = new_critic_obs_dim
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(act_fn())
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        # Store dimensions for reference
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions

    def reset(self, dones=None):
        """Reset the distribution (called at episode boundaries)."""
        pass

    def forward(self):
        """Not implemented - use act() or evaluate() instead."""
        raise NotImplementedError

    def _encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode path slice and concatenate with other observations.

        Args:
            observations: Full observation tensor (batch, obs_dim)

        Returns:
            Encoded observations (batch, new_obs_dim)
        """
        # Split path slice from other observations
        path_slice = observations[:, :self.path_slice_dim]
        other_obs = observations[:, self.path_slice_dim:]

        # Encode path slice
        encoded_path = self.path_encoder(path_slice)

        # Concatenate encoded path with other observations
        return torch.cat([encoded_path, other_obs], dim=-1)

    @property
    def action_mean(self):
        """Return the mean of the action distribution."""
        return self.distribution.mean

    @property
    def action_std(self):
        """Return the standard deviation of the action distribution."""
        return self.distribution.stddev

    @property
    def entropy(self):
        """Return the entropy of the action distribution."""
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor):
        """Update the action distribution based on observations."""
        encoded_obs = self._encode_observations(observations)
        mean = self.actor(encoded_obs)
        self.distribution = torch.distributions.Normal(mean, self.std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample actions from the policy.

        Args:
            observations: Observation tensor (batch, obs_dim)

        Returns:
            Sampled actions (batch, num_actions)
        """
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probability of actions under current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Get deterministic actions for inference.

        Args:
            observations: Observation tensor (batch, obs_dim)

        Returns:
            Mean actions (batch, num_actions)
        """
        encoded_obs = self._encode_observations(observations)
        return self.actor(encoded_obs)

    def evaluate(
        self, critic_observations: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Evaluate the value function.

        Args:
            critic_observations: Critic observation tensor (batch, critic_obs_dim)

        Returns:
            Value estimates (batch, 1)
        """
        encoded_obs = self._encode_observations(critic_observations)
        return self.critic(encoded_obs)
