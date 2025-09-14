from __future__ import annotations

from pathlib import Path
import json
import csv
import time
from typing import Any, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack


def eval(
    model: PPO,
    config: Any,
    n_eval_episodes: int = 10,
    deterministic: Optional[bool] = None,
    save_dir: Optional[str] = None,
    tag: Optional[str] = None,
    record_logger: bool = True,
) -> dict:
    """Evaluate a trained PPO model and optionally persist metrics.

    Parameters
    ----------
    model : PPO
        Trained model to evaluate.
    config : object
        Config object (AtariConfig / MujocoConfig). Must expose: env_id, atari_env, seed.
    n_eval_episodes : int, default 10
        Number of evaluation episodes.
    deterministic : bool | None
        Whether to use deterministic actions. If None: Atari -> False, Mujoco -> True.
    save_dir : str | None
        Directory to which metrics (jsonl / csv / latest txt) are written. Defaults to
        config.tensorboard_log / 'eval'.
    tag : str | None
        Tag used in metric names & filenames (defaults: 'atari' or 'mujoco').
    record_logger : bool
        If True and model has a logger, record metrics to it.

    Returns
    -------
    dict
        Dict containing mean_reward, std_reward, episodes and elapsed_seconds.
    """

    is_atari = bool(getattr(config, "atari_env", False))
    if deterministic is None:
        deterministic = not is_atari  # Atari often stochastic policy during eval
    if tag is None:
        tag = "atari" if is_atari else "mujoco"

    # Build evaluation environment (independent from training env)
    if is_atari:
        n_stack = getattr(config, "n_stack", 4)
        env = make_atari_env(config.env_id, n_envs=1, seed=getattr(config, "seed", None))
        env = VecFrameStack(env, n_stack=n_stack)
    else:
        env = make_vec_env(config.env_id, n_envs=1, seed=getattr(config, "seed", None))

    start = time.time()
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=False,
    )
    elapsed = time.time() - start

    env.close()

    # Record to SB3 logger
    if record_logger and hasattr(model, "logger"):
        model.logger.record(f"eval/{tag}_mean_reward", mean_reward)
        model.logger.record(f"eval/{tag}_std_reward", std_reward)
        model.logger.record(f"eval/{tag}_episodes", n_eval_episodes)
        model.logger.record(f"eval/{tag}_seconds", elapsed)
        model.logger.dump(model.num_timesteps)

    # Persist to files
    if save_dir is None:
        base = getattr(config, "tensorboard_log", "logs")
        save_dir = str(Path(base) / "eval")
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON Lines
    jsonl_path = out_dir / f"{tag}_reward_log.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        record = {
            "tag": tag,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "episodes": n_eval_episodes,
            "timesteps": model.num_timesteps,
            "seconds": elapsed,
            "env_id": getattr(config, "env_id", None),
            "seed": getattr(config, "seed", None),
        }
        f.write(json.dumps(record) + "\n")

    # CSV (append)
    csv_path = out_dir / f"{tag}_reward_log.csv"
    new_csv = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_csv:
            writer.writerow([
                "tag",
                "env_id",
                "timesteps",
                "episodes",
                "mean_reward",
                "std_reward",
                "seconds",
                "seed",
            ])
        writer.writerow([
            tag,
            getattr(config, "env_id", None),
            model.num_timesteps,
            n_eval_episodes,
            mean_reward,
            std_reward,
            elapsed,
            getattr(config, "seed", None),
        ])

    # Latest (overwrite)
    latest_path = out_dir / f"{tag}_latest.txt"
    latest_path.write_text(
        f"{tag} mean_reward={mean_reward:.3f} std={std_reward:.3f} episodes={n_eval_episodes} steps={model.num_timesteps} elapsed_s={elapsed:.1f}\n",
        encoding="utf-8",
    )

    return dict(
        mean_reward=mean_reward,
        std_reward=std_reward,
        episodes=n_eval_episodes,
        seconds=elapsed,
        tag=tag,
    )