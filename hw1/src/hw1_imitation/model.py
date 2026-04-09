"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dims
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(self.state_dim, self.hidden_dim[0]))
        self.mlp.append(nn.ReLU())
        for i in range(len(self.hidden_dim) - 1):
            self.mlp.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i+1]))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(self.hidden_dim[-1], self.chunk_size * self.action_dim))
        self.mlp = nn.Sequential(*self.mlp)
        self.loss = nn.MSELoss()

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # raise NotImplementedError
        #action chunk b, chunk_size, dim
        pred = self.mlp(state).reshape(*action_chunk.shape)
        return self.loss(pred, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # raise NotImplementedError
        return self.mlp(state).reshape(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dims
        self.input_dim = self.state_dim + self.chunk_size * self.action_dim + 1
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(self.input_dim, self.hidden_dim[0]))
        self.mlp.append(nn.ReLU())
        for i in range(len(self.hidden_dim) - 1):
            self.mlp.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i+1]))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(self.hidden_dim[-1], self.chunk_size * self.action_dim))
        self.mlp = nn.Sequential(*self.mlp)
        self.loss = nn.MSELoss()

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # raise NotImplementedError
        t = torch.rand([state.shape[0], 1]).to(dtype=state.dtype, device=state.device) #B
        action_chunk = action_chunk.reshape(-1, self.chunk_size * self.action_dim)
        noise = torch.randn_like(action_chunk).to(dtype=state.dtype, device=state.device)
        a_t = t * action_chunk + (1 - t) * noise #bs, chunk * dim
        in_x = torch.concat([state, a_t, t], dim=-1)
        return self.loss((action_chunk - noise), self.mlp(in_x))

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # raise NotImplementedError
        prev_a = torch.randn(state.shape[0], self.chunk_size * self.action_dim).to(dtype=state.dtype, device=state.device)
        dt = 1 / num_steps
        t = torch.tensor(0).to(dtype=state.dtype, device=state.device).repeat([state.shape[0]]).unsqueeze(1)
        for i in range(num_steps):
            in_x = torch.concat([state, prev_a, t],dim=-1)
            prev_a = prev_a + dt * self.mlp(in_x)
            t += dt
        return prev_a.reshape([-1, self.chunk_size, self.action_dim])


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
