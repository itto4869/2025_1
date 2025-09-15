from .config import MujocoConfig, AtariConfig
from .train import train
from .network import MujocoPolicy, AtariPolicy
from .optimizer import create_optimizer_kwargs, atari_adam_lr_schedule, atari_soap_lr_schedule
from .eval import eval
from stable_baselines3 import PPO
import time

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
        match optimizer_name:
            case "adam":
                learning_rate = atari_adam_lr_schedule
            case "soap":
                learning_rate = atari_soap_lr_schedule
            case "muon":
                learning_rate = atari_adam_lr_schedule
        return AtariConfig(
            env_id=env_id,
            learning_rate=learning_rate,
            policy_model=AtariPolicy,
            optimizer_kwargs=optimizer_kwargs,
            tensorboard_log=tensorboard_log,
            seed=seed,
            total_timesteps=total_timesteps,
            save_path=save_path,
        )
    else:
        match optimizer_name:
            case "adam":
                learning_rate = 3e-4
            case "soap":
                learning_rate = 3e-3
            case "muon":
                learning_rate = 2e-2
        return MujocoConfig(
            env_id=env_id,
            learning_rate=learning_rate,
            policy_model=MujocoPolicy,
            optimizer_kwargs=optimizer_kwargs,
            tensorboard_log=tensorboard_log,
            seed=seed,
            total_timesteps=total_timesteps,
            save_path=save_path,
        )

def run_experiment(config: AtariConfig | MujocoConfig) -> tuple[PPO, float]:
    start = time.perf_counter()
    model = train(config)
    end = time.perf_counter()
    print(f"Training time: {end - start:.2f} seconds")
    duration = end - start
    # TensorBoard logging of wall-clock and throughput
    try:
        steps = getattr(model, "num_timesteps", None)
        if hasattr(model, "logger"):
            model.logger.record("time/training_wall_clock_sec", float(duration))
            if isinstance(steps, int) and duration > 0:
                model.logger.record("time/steps_per_second", float(steps) / duration)
            # dump at final step (use steps or 0 fallback)
            dump_step = steps if isinstance(steps, int) else 0
            model.logger.dump(step=dump_step)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] failed to log wall clock to TensorBoard: {e}")
    return model, duration
    
if __name__ == "__main__":
    seeds = [7, 19, 801, 3, 2025, 777, 1015, 420, 906, 75]
    optimizer_names = ["soap"]
    for seed in seeds:
        for optimizer_name in optimizer_names:
            atari_config = set_config(
                env_id="ale_py:ALE/Breakout-v5",
                optimizer_name=optimizer_name,
                tensorboard_log="tensorboard_logs/atari/{optimizer_name}/{seed}".format(seed=seed, optimizer_name=optimizer_name),
                seed=seed,
                total_timesteps=10_000_000,
                atari=True,
                save_path="models/atari/{optimizer_name}/{seed}".format(seed=seed, optimizer_name=optimizer_name),
            )
            
            mujoco_config = set_config(
                env_id="HalfCheetah-v5",
                optimizer_name=optimizer_name,
                tensorboard_log="tensorboard_logs/mujoco/{optimizer_name}/{seed}".format(seed=seed, optimizer_name=optimizer_name),
                seed=seed,
                total_timesteps=5_000_000,
                atari=False,
                save_path="models/mujoco/{optimizer_name}/{seed}".format(seed=seed, optimizer_name=optimizer_name),
            )
            
            atari_model, atari_wall_clock = run_experiment(atari_config)
            #mujoco_model, mujoco_wall_clock = run_experiment(mujoco_config)

            atari_metrics = eval(atari_model, atari_config)
            #mujoco_metrics = eval(mujoco_model, mujoco_config)

            print(f"Seed: {seed}, Optimizer: {optimizer_name}")
            print(f"Atari Metrics: {atari_metrics}")
            #print(f"Mujoco Metrics: {mujoco_metrics}")