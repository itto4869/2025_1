from typing import Callable

import torch
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from muon import SingleDeviceMuonWithAuxAdam

class MLPExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64
    ) -> None:
        super().__init__()
        
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.Tanh(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi), nn.Tanh(),
        )

        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.Tanh(),
            nn.Linear(last_layer_dim_vf, last_layer_dim_vf), nn.Tanh(),
        )
    
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

class MujocoPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ) -> None:
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MLPExtractor(self.features_dim)  # type: ignore

class MujocoMuonPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ) -> None:
        muon_cfg = kwargs.pop("muon_cfg", {})
        self.muon_lr = muon_cfg.get("muon_lr", 2e-2)
        self.muon_weight_decay = muon_cfg.get("muon_weight_decay", 1e-2)

        self.adam_betas = muon_cfg.get("adam_betas", (0.9, 0.95))
        self.adam_weight_decay = muon_cfg.get("adam_weight_decay", 1e-2)

        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MLPExtractor(self.features_dim)  # type: ignore

    def _setup_optimizer(self) -> None:
        hidden_modules = []
        if getattr(self, "features_extractor", None) is not None:
            hidden_modules.append(self.features_extractor)
        if getattr(self, "mlp_extractor", None) is not None:
            hidden_modules.append(self.mlp_extractor) # type: ignore
        
        hidden_weights, hidden_gains_biases = [], []
        for mod in hidden_modules:
            for p in mod.parameters():
                if not p.requires_grad:
                    continue
                (hidden_weights if p.ndim >= 2 else hidden_gains_biases).append(p)
        
        nonhidden_params = list(self.action_net.parameters()) + list(self.value_net.parameters())
        
        adam_lr = float(self.lr_schedule(1.0)) # type: ignore
        
        param_groups = []
        if len(hidden_weights) > 0:
            param_groups.append(dict(params=hidden_weights, use_muon=True,
                                    lr=self.muon_lr, weight_decay=self.muon_weight_decay)) # type: ignore
        if len(hidden_gains_biases) + len(nonhidden_params) > 0:
            param_groups.append(dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
                                    lr=adam_lr, betas=self.adam_betas, weight_decay=self.adam_weight_decay))
        
        self.optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

class CNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
    
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU(),
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class AtariPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ) -> None:
        kwargs["features_extractor_class"] = CNNExtractor
        kwargs["features_extractor_kwargs"] = dict(features_dim=256)
    
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

class AtariMuonPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ) -> None:
        muon_cfg = kwargs.pop("muon_cfg", {})
        self.muon_lr = muon_cfg.get("muon_lr", 2e-2)
        self.muon_weight_decay = muon_cfg.get("muon_weight_decay", 1e-2)

        self.adam_betas = muon_cfg.get("adam_betas", (0.9, 0.95))
        self.adam_weight_decay = muon_cfg.get("adam_weight_decay", 1e-2)

        kwargs["features_extractor_class"] = CNNExtractor
        kwargs["features_extractor_kwargs"] = dict(features_dim=256)
        
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
    
    def _setup_optimizer(self) -> None:
        hidden_modules = []
        if getattr(self, "features_extractor", None) is not None:
            hidden_modules.append(self.features_extractor)
        if getattr(self, "mlp_extractor", None) is not None:
            hidden_modules.append(self.mlp_extractor) # type: ignore
        
        hidden_weights, hidden_gains_biases = [], []
        for mod in hidden_modules:
            for p in mod.parameters():
                if not p.requires_grad:
                    continue
                (hidden_weights if p.ndim >= 2 else hidden_gains_biases).append(p)
        
        nonhidden_params = list(self.action_net.parameters()) + list(self.value_net.parameters())
        
        adam_lr = float(self.lr_schedule(1.0)) # type: ignore
        
        param_groups = []
        if len(hidden_weights) > 0:
            param_groups.append(dict(params=hidden_weights, use_muon=True,
                                    lr=self.muon_lr, weight_decay=self.muon_weight_decay)) # type: ignore
        if len(hidden_gains_biases) + len(nonhidden_params) > 0:
            param_groups.append(dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
                                    lr=adam_lr, betas=self.adam_betas, weight_decay=self.adam_weight_decay))
            
        self.optimizer = SingleDeviceMuonWithAuxAdam(param_groups)