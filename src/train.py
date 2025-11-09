from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from typing import Any
from muon import SingleDeviceMuonWithAuxAdam
import torch
import random
import numpy as np
from types import MethodType
from .callbacks import UpdateMagnitudeCallback

def train(config) -> PPO:
    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)

    # Create the environment
    if config.atari_env:
        env = make_atari_env(config.env_id, n_envs=config.n_envs, seed=config.seed, vec_env_cls=SubprocVecEnv, env_kwargs=config.env_kwargs)
        env = VecFrameStack(env, n_stack=config.n_stack)
    else:
        env = make_vec_env(config.env_id, n_envs=config.n_envs, seed=config.seed, vec_env_cls=SubprocVecEnv)

    # Initialize the PPO model
    if "muon_cfg" in config.optimizer_kwargs:
        model = PPO(
            policy=config.policy_model,
            env=env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            max_grad_norm=config.max_grad_norm,
            normalize_advantage=config.normalize_advantage,
            verbose=config.verbose,
            tensorboard_log=config.tensorboard_log,
            seed=config.seed,
        )
    
    else:
        model = PPO(
            policy=config.policy_model,
            env=env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            max_grad_norm=config.max_grad_norm,
            normalize_advantage=config.normalize_advantage,
            verbose=config.verbose,
            tensorboard_log=config.tensorboard_log,
            seed=config.seed,
            policy_kwargs=config.optimizer_kwargs,
        )

    muon_cfg: dict[str, Any] | None = None
    if isinstance(getattr(config, 'optimizer_kwargs', None), dict):
        if 'muon_cfg' in config.optimizer_kwargs:
            muon_cfg = config.optimizer_kwargs['muon_cfg']  # type: ignore[index]
    if muon_cfg and SingleDeviceMuonWithAuxAdam is not None:
        pol = model.policy  # ActorCriticPolicy instance
        try:
            muon_lr = float(muon_cfg.get('muon_lr', 2e-2))
            muon_wd = float(muon_cfg.get('muon_weight_decay', 1e-2))
            adam_betas = muon_cfg.get('adam_betas', (0.9, 0.95))
            adam_wd = float(muon_cfg.get('adam_weight_decay', 1e-2))
            # Use 0.0 floor by default to allow pure linear decay when desired
            muon_lr_floor = float(muon_cfg.get('muon_lr_floor', 0.0))
            # Derive Adam lr from existing optimizer first group or fallback
            old_opt = getattr(pol, 'optimizer', None)
            if old_opt is not None and getattr(old_opt, 'param_groups', None):
                adam_lr = float(old_opt.param_groups[0].get('lr', 3e-4))  # type: ignore[index]
            else:
                # If schedule was constant numeric value
                if isinstance(config.learning_rate, (int, float)):
                    adam_lr = float(config.learning_rate)
                else:
                    adam_lr = 3e-4

            # Collect params by module for finer-grained control
            fe_weights, fe_bias = [], []
            mlp_weights, mlp_bias = [], []
            if hasattr(pol, 'features_extractor'):
                for p in pol.features_extractor.parameters():
                    if not p.requires_grad:
                        continue
                    (fe_weights if p.ndim >= 2 else fe_bias).append(p)
            if hasattr(pol, 'mlp_extractor'):
                for p in pol.mlp_extractor.parameters():  # type: ignore[attr-defined]
                    if not p.requires_grad:
                        continue
                    (mlp_weights if p.ndim >= 2 else mlp_bias).append(p)

            head_params = list(pol.action_net.parameters()) + list(pol.value_net.parameters())

            # Apply Muon to both CNN (features_extractor) and MLP weights
            muon_params = fe_weights + mlp_weights

            # Build optimizer param groups
            param_groups: list[dict[str, Any]] = []
            if muon_params:
                param_groups.append(dict(params=muon_params, use_muon=True,
                                         lr=muon_lr, weight_decay=muon_wd))
            # Keep biases and policy/value heads on Adam
            adam_params_all = fe_bias + mlp_bias + head_params
            
            if hasattr(pol, "log_std") and isinstance(pol.log_std, torch.nn.Parameter):
                if pol.log_std.requires_grad:
                    adam_params_all.append(pol.log_std)
            if adam_params_all:
                param_groups.append(dict(params=adam_params_all, use_muon=False,
                                         lr=adam_lr, betas=adam_betas, weight_decay=adam_wd))
            # Replace optimizer
            pol.optimizer = SingleDeviceMuonWithAuxAdam(param_groups)  # type: ignore[call-arg]
            print(f"[muon] Replaced optimizer with hybrid Muon+Adam: groups={len(param_groups)} muon_lr={muon_lr} adam_lr={adam_lr}")

            # Make Muon follow SB3 LR schedule shape (ratio-coupled) with a floor,
            # preserving its own base scale. Override the algorithm-level hook so it is used.
            def _update_learning_rate(self, optimizers,
                                      _muon_lr_base=muon_lr, _adam_lr_base=adam_lr,
                                      _muon_lr_floor=muon_lr_floor):
                # Normalize to list
                if not isinstance(optimizers, (list, tuple)):
                    optimizers = [optimizers]
                # Compute base LR from SB3 schedule (progress 1->0)
                lr = self.lr_schedule(self._current_progress_remaining)  # type: ignore[attr-defined]
                for optimizer in optimizers:
                    for group in optimizer.param_groups:
                        if group.get('use_muon', False):
                            if _adam_lr_base > 0:
                                scaled = _muon_lr_base * (lr / _adam_lr_base)
                            else:
                                scaled = _muon_lr_base
                            group['lr'] = max(_muon_lr_floor, float(scaled))
                        else:
                            group['lr'] = lr
            # Patch PPO/BaseAlgorithm instance, not the policy
            model._update_learning_rate = MethodType(_update_learning_rate, model)
        except Exception as e:  # noqa: BLE001
            # Print full traceback for debugging
            import traceback
            print(f"[muon] Hybrid optimizer replacement failed: {e}\n{traceback.format_exc()}")
    elif muon_cfg and SingleDeviceMuonWithAuxAdam is None:
        print("[muon] muon library not available; falling back to original optimizer")

    # Train the agent with update-magnitude logging
    model.learn(total_timesteps=config.total_timesteps, progress_bar=True,
                callback=UpdateMagnitudeCallback())
    
    # Save the model
    if config.save_path is not None:
        torch.save(model.policy.state_dict(), config.save_path + "policy.pth")
        #model.save(config.save_path)

    return model
