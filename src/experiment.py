import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO

from .config import (
    DEFAULT_ATARI_BATCH_MULTIPLIER,
    DEFAULT_ATARI_N_ENVS,
    AtariConfig,
    MujocoConfig,
)
from .eval import eval
from .network import AtariPolicy, MujocoPolicy
from .optimizer import (
    atari_adam_lr_schedule,
    atari_soap_lr_schedule,
    create_optimizer_kwargs,
)
from .train import train


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

MUJOCO_ENV_IDS = {
    "ant": "Ant-v5",
    "halfcheetah": "HalfCheetah-v5",
    "hopper": "Hopper-v5",
    "walker2d": "Walker2d-v5",
    "humanoid": "Humanoid-v5",
    "humanoidstandup": "HumanoidStandup-v5",
    "invertedpendulum": "InvertedPendulum-v5",
    "inverteddoublependulum": "InvertedDoublePendulum-v5",
    "swimmer": "Swimmer-v5",
    "reacher": "Reacher-v5",
    "pusher": "Pusher-v5",
}

PARALLEL_EXPERIMENTS = 2
ATARI_TOTAL_TIMESTEPS = 10_000_000
MUJOCO_TOTAL_TIMESTEPS = 1_000_000
_MP_CONTEXT = multiprocessing.get_context("spawn")


@dataclass(frozen=True)
class ExperimentJob:
    suite: str  # "atari" or "mujoco"
    seed: int
    optimizer: str
    env_key: str
    env_id: str
    total_timesteps: int


def _ensure_keys(requested: list[str] | None, mapping: dict[str, str], suite: str) -> list[str]:
    if requested is None:
        return list(mapping.keys())
    missing = [key for key in requested if key not in mapping]
    if missing:
        raise ValueError(f"Unknown {suite} env keys: {', '.join(missing)}")
    return requested

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
                learning_rate = 3e-4
            case "muon":
                learning_rate = 3e-4
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


def _run_single_job(job: ExperimentJob) -> dict:
    is_atari = job.suite == "atari"
    tensorboard_dir = f"tensorboard_logs/{job.suite}/{job.env_key}/{job.optimizer}/{job.seed}"
    model_dir = f"models/{job.suite}/{job.env_key}/{job.optimizer}/{job.seed}"
    Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    config_kwargs = dict(
        env_id=job.env_id,
        optimizer_name=job.optimizer,
        tensorboard_log=tensorboard_dir,
        seed=job.seed,
        total_timesteps=job.total_timesteps,
        atari=is_atari,
        save_path=f"{model_dir}/",
    )
    if is_atari:
        config_kwargs["n_envs"] = DEFAULT_ATARI_N_ENVS
    config = set_config(**config_kwargs)

    model, wall_clock = run_experiment(config)
    metrics = eval(model, config, tag=job.env_key)

    try:
        if hasattr(model, "env"):
            model.env.close()  # type: ignore[call-arg]
    except Exception:  # noqa: BLE001
        pass

    del model
    return {
        "seed": job.seed,
        "optimizer": job.optimizer,
        "suite": job.suite,
        "env_key": job.env_key,
        "env_id": job.env_id,
        "training_time": wall_clock,
        "metrics": metrics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Atari and Mujoco training suites.")
    parser.add_argument(
        "--suite",
        choices=["atari", "mujoco", "both"],
        default="atari",
        help="Choose which suite to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 19, 801, 3, 2025, 777, 1015, 420, 906, 75],
        help="Random seeds to sweep.",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["muon"],
        help="Optimizer names to evaluate.",
    )
    parser.add_argument(
        "--atari-envs",
        nargs="+",
        default=None,
        help="Subset of Atari env keys to run.",
    )
    parser.add_argument(
        "--mujoco-envs",
        nargs="+",
        default=None,
        help="Subset of Mujoco env keys to run.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=PARALLEL_EXPERIMENTS,
        help="Maximum concurrent experiments.",
    )
    parser.add_argument(
        "--atari-timesteps",
        type=int,
        default=ATARI_TOTAL_TIMESTEPS,
        help="Total timesteps per Atari run.",
    )
    parser.add_argument(
        "--mujoco-timesteps",
        type=int,
        default=MUJOCO_TOTAL_TIMESTEPS,
        help="Total timesteps per Mujoco run.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()

    suites: list[str]
    if args.suite == "both":
        suites = ["atari", "mujoco"]
    else:
        suites = [args.suite]

    atari_keys = (
        _ensure_keys(args.atari_envs, ATARI_ENV_IDS, "atari") if "atari" in suites else []
    )
    mujoco_keys = (
        _ensure_keys(args.mujoco_envs, MUJOCO_ENV_IDS, "mujoco") if "mujoco" in suites else []
    )

    jobs: list[ExperimentJob] = []
    for suite in suites:
        env_keys = atari_keys if suite == "atari" else mujoco_keys
        timestep_budget = args.atari_timesteps if suite == "atari" else args.mujoco_timesteps
        for seed in args.seeds:
            for optimizer_name in args.optimizers:
                for env_key in env_keys:
                    env_id = ATARI_ENV_IDS[env_key] if suite == "atari" else MUJOCO_ENV_IDS[env_key]
                    jobs.append(
                        ExperimentJob(
                            suite=suite,
                            seed=seed,
                            optimizer=optimizer_name,
                            env_key=env_key,
                            env_id=env_id,
                            total_timesteps=timestep_budget,
                        )
                    )

    if not jobs:
        raise RuntimeError("No experiments scheduled. Check suite and env selections.")

    with ProcessPoolExecutor(max_workers=args.max_workers, mp_context=_MP_CONTEXT) as executor:
        futures = {executor.submit(_run_single_job, job): job for job in jobs}

        for future in as_completed(futures):
            result = future.result()
            metrics = result["metrics"]
            print(
                "[{suite}] Seed: {seed}, Optimizer: {optimizer}, Env: {env}, Time: {time:.2f}s".format(
                    suite=result["suite"],
                    seed=result["seed"],
                    optimizer=result["optimizer"],
                    env=result["env_id"],
                    time=result["training_time"],
                )
            )
            print(f"Metrics ({result['suite']}:{result['env_key']}): {metrics}")
