from dataclasses import dataclass, field
from typing import Callable
from stable_baselines3.common.policies import ActorCriticPolicy
# TODO: パラメータは要チェック
def atari_lr_schedule(progress: float) -> float:
    return 2.5e-4 * progress

def clip_range_schedule(progress: float) -> float:
    return 0.1 * progress

@dataclass
class AtariConfig:
    env_id: str
    policy_model: type[ActorCriticPolicy]
    optimizer_kwargs: dict
    tensorboard_log: str
    atari_env: bool = True
    n_envs: int = 8
    n_stack: int = 4
    n_steps: int = 128
    learning_rate: Callable[[float], float] = atari_lr_schedule
    n_epochs: int = 3
    batch_size: int = 32 * 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: Callable[[float], float] | float = clip_range_schedule
    vf_coef: float = 1
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    verbose: int = 1
    total_timesteps: int = 10_000_000
    n_eval_episodes: int = 100
    env_kwargs: dict | None = field(default_factory=lambda: {"frameskip": 1})
    save_path: str | None = None
    seed: int | None = None

@dataclass
class MujocoConfig:
    env_id: str
    policy_model: type[ActorCriticPolicy]
    optimizer_kwargs: dict
    tensorboard_log: str
    atari_env: bool = False
    n_envs: int = 8
    n_steps: int = 2048
    learning_rate: Callable[[float], float] | float = 3e-4
    n_epochs: int = 10
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: Callable[[float], float] | float = 0.2
    vf_coef: float = 0.0
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    verbose: int = 1
    total_timesteps: int = 1_000_000
    n_eval_episodes: int = 100
    save_path: str | None = None
    seed: int | None = None