import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .config import (
    DEFAULT_ATARI_BATCH_MULTIPLIER,
    DEFAULT_ATARI_N_ENVS,
    AtariConfig,
    MujocoConfig,
)
from .train import train
from .network import MujocoPolicy, AtariPolicy
from .optimizer import create_optimizer_kwargs, atari_adam_lr_schedule, atari_soap_lr_schedule
from .eval import eval
from stable_baselines3 import PPO
import time


ATARI_ENV_IDS = {
    "breakout": "ale_py:ALE/Breakout-v5",
    "space_invaders": "ale_py:ALE/SpaceInvaders-v5",
    "beam_rider": "ale_py:ALE/BeamRider-v5",
    "freeway": "ale_py:ALE/Freeway-v5",
    "seaquest": "ale_py:ALE/Seaquest-v5",
    "enduro": "ale_py:ALE/Enduro-v5",
    "frostbite": "ale_py:ALE/Frostbite-v5",
    "ms_pacman": "ale_py:ALE/MsPacman-v5",
    "hero": "ale_py:ALE/Hero-v5",
    "montezuma_revenge": "ale_py:ALE/MontezumaRevenge-v5",
}

PARALLEL_EXPERIMENTS = 2
ATARI_TOTAL_TIMESTEPS = 10_000_000
_MP_CONTEXT = multiprocessing.get_context("spawn")

def set_config(
    env_id: str,
    optimizer_name: str,
    tensorboard_log: str,
    atari: bool,
    seed: int | None = None,
    total_timesteps: int = 1_000_000,
    save_path: str | None = None,
    n_envs: int | None = None,
    batch_size: int | None = None,
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
        resolved_n_envs = n_envs if n_envs is not None else DEFAULT_ATARI_N_ENVS
        resolved_batch_size = (
            batch_size
            if batch_size is not None
            else DEFAULT_ATARI_BATCH_MULTIPLIER * resolved_n_envs
        )
        return AtariConfig(
            env_id=env_id,
            learning_rate=learning_rate,
            policy_model=AtariPolicy,
            optimizer_kwargs=optimizer_kwargs,
            tensorboard_log=tensorboard_log,
            seed=seed,
            total_timesteps=total_timesteps,
            save_path=save_path,
            n_envs=resolved_n_envs,
            batch_size=resolved_batch_size,
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


def _run_single_atari_experiment(
    seed: int,
    optimizer_name: str,
    game_key: str,
    env_id: str,
) -> dict:
    tensorboard_dir = f"tensorboard_logs/atari/{game_key}/{optimizer_name}/{seed}"
    model_dir = f"models/atari/{game_key}/{optimizer_name}/{seed}"
    Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    config = set_config(
        env_id=env_id,
        optimizer_name=optimizer_name,
        tensorboard_log=tensorboard_dir,
        seed=seed,
        total_timesteps=ATARI_TOTAL_TIMESTEPS,
        atari=True,
        save_path=f"{model_dir}/",
        n_envs=DEFAULT_ATARI_N_ENVS,
    )

    model, wall_clock = run_experiment(config)
    metrics = eval(model, config, tag=game_key)

    # Release vectorized environment resources before returning to parent.
    try:
        if hasattr(model, "env"):
            model.env.close()  # type: ignore[call-arg]
    except Exception:  # noqa: BLE001
        pass

    del model
    return {
        "seed": seed,
        "optimizer": optimizer_name,
        "game_key": game_key,
        "env_id": env_id,
        "training_time": wall_clock,
        "metrics": metrics,
    }


if __name__ == "__main__":
    seeds = [7, 19, 801, 3, 2025, 777, 1015, 420, 906, 75]
    optimizer_names = ["muon"]

    jobs: list[tuple[int, str, str, str]] = []
    for seed in seeds:
        for optimizer_name in optimizer_names:
            for game_key, env_id in ATARI_ENV_IDS.items():
                jobs.append((seed, optimizer_name, game_key, env_id))

    with ProcessPoolExecutor(max_workers=PARALLEL_EXPERIMENTS, mp_context=_MP_CONTEXT) as executor:
        futures = {
            executor.submit(_run_single_atari_experiment, seed, optimizer, game_key, env_id):
            (seed, optimizer, game_key, env_id)
            for seed, optimizer, game_key, env_id in jobs
        }

        for future in as_completed(futures):
            result = future.result()
            metrics = result["metrics"]
            print(
                "Seed: {seed}, Optimizer: {optimizer}, Game: {game}, Time: {time:.2f}s".format(
                    seed=result["seed"],
                    optimizer=result["optimizer"],
                    game=result["env_id"],
                    time=result["training_time"],
                )
            )
            print(f"Metrics ({result['game_key']}): {metrics}")
