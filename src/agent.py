import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _freeze_module(module: nn.Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False


def linear_schedule(start_e: float, end_e: float, duration: float, t: int) -> float:
    """Linear annealing schedule for epsilon-greedy."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class NatureCNN(nn.Module):
    """CNN feature extractor from the DQN nature paper."""

    def __init__(self, double_width: bool) -> None:
        super().__init__()

        self.double_width = double_width
        self.output_dims = 1024 if self.double_width else 512

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, self.output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x / 255.0 - 0.5))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return x


class InFeRHead(nn.Module):
    """Minimal InFeR head."""

    def __init__(self, feat_size: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(feat_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x)


class InFerDDQN(nn.Module):
    """InFeR DDQN agent."""

    def __init__(self, env, use_infer: bool, num_heads: int, scaling_factor: float, double_width: bool):
        super().__init__()

        self.use_infer = use_infer
        self.num_heads = num_heads
        self.scaling_factor = scaling_factor
        self.double_width = double_width

        self.linear_feat_size = 1024 if self.double_width else 512

        self.phi = NatureCNN(self.double_width)
        if self.use_infer:
            # Copy initial network parameters and freeze them
            self.phi0 = copy.deepcopy(self.phi)
            _freeze_module(self.phi0)

            # Initialize the InFeR heads
            self.infer_heads = nn.ModuleList([InFeRHead(self.linear_feat_size) for _ in range(self.num_heads)])
            self.infer_target_heads = copy.deepcopy(self.infer_heads)
            for head in self.infer_target_heads:
                _freeze_module(head)

        self.q = nn.Linear(self.linear_feat_size, int(env.single_action_space.n))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        train_feats = self.phi(x)
        q_vals = self.q(train_feats)
        return q_vals, train_feats

    def get_infer_vals(self, x: torch.Tensor, train_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # Use features of initial network to compute targets for InFeR
            infer_feats = self.phi0(x)
            infer_targets = torch.cat([target_head(infer_feats) for target_head in self.infer_target_heads], dim=1)

        infer_preds = torch.cat([infer_head(train_feats) for infer_head in self.infer_heads], dim=1)

        return infer_preds, self.scaling_factor * infer_targets
