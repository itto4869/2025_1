from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
import torch
import random
import numpy as np

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

    # Train the agent
    model.learn(total_timesteps=config.total_timesteps, progress_bar=True)
    
    # Save the model
    if config.save_path is not None:
        model.save(config.save_path)

    return model