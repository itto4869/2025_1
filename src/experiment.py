from .config import MujocoConfig, AtariConfig
from .train import train
from .network import MujocoPolicy, MujocoMuonPolicy, AtariPolicy, AtariMuonPolicy
from .optimizer import create_optimizer_kwargs
from .eval import eval
from stable_baselines3 import PPO

def set_config(
    env_id: str,
    optimizer_name: str,
    tensorboard_log: str,
    atari: bool,
    seed: int | None = None,
    total_timesteps: int = 1_000_000,
    save_path: str | None = None,
) -> AtariConfig | MujocoConfig:
    optimizer_kwargs = create_optimizer_kwargs(optimizer_name)
    if atari:
        if optimizer_name == "muon":
            policy_model = AtariMuonPolicy
        else:
            policy_model = AtariPolicy
        return AtariConfig(
            env_id=env_id,
            policy_model=policy_model,
            optimizer_kwargs=optimizer_kwargs,
            tensorboard_log=tensorboard_log,
            seed=seed,
            total_timesteps=total_timesteps,
            save_path=save_path,
        )
    else:
        if optimizer_name == "muon":
            policy_model = MujocoMuonPolicy
        else:
            policy_model = MujocoPolicy
        return MujocoConfig(
            env_id=env_id,
            policy_model=policy_model,
            optimizer_kwargs=optimizer_kwargs,
            tensorboard_log=tensorboard_log,
            seed=seed,
            total_timesteps=total_timesteps,
            save_path=save_path,
        )

def run_experiment(config: AtariConfig | MujocoConfig) -> PPO:
    model = train(config)
    return model
    
if __name__ == "__main__":
    seed = 42
    optimizer_name = "muon"
    mujoco_config = set_config(
        env_id="HalfCheetah-v5",
        optimizer_name=optimizer_name,
        tensorboard_log="tensorboard_logs/{seed}/mujoco/{optimizer_name}".format(seed=seed, optimizer_name=optimizer_name),
        seed=seed,
        total_timesteps=1_000_000,
        atari=False,
        save_path="models/{seed}/mujoco/{optimizer_name}".format(seed=seed, optimizer_name=optimizer_name),
    )

    atari_config = set_config(
        env_id="ale_py:ALE/Breakout-v5",
        optimizer_name=optimizer_name,
        tensorboard_log="tensorboard_logs/{seed}/atari/{optimizer_name}".format(seed=seed, optimizer_name=optimizer_name),
        seed=seed,
        total_timesteps=1_000_000,
        atari=True,
        save_path="models/{seed}/atari/{optimizer_name}".format(seed=seed, optimizer_name=optimizer_name),
    )
    
    mujoco_model = run_experiment(mujoco_config)
    atari_model = run_experiment(atari_config)
    
    mujoco_mean_reward = eval(mujoco_model, mujoco_config)
    atari_mean_reward = eval(atari_model, atari_config)
    
    print(f"Mujoco Mean Reward: {mujoco_mean_reward}")
    print(f"Atari Mean Reward: {atari_mean_reward}")