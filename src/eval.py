from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

def eval(model: PPO, config):
    """
    Evaluate the trained model.

    Args:
        model (PPO): The trained PPO model.
        n_eval_episodes (int): Number of episodes to evaluate.

    Returns:
        float: Average reward over the evaluation episodes.
    """
    if config.atari_env:
        env = make_atari_env(config.env_id, n_envs=1, seed=config.seed)
        env = VecFrameStack(env, n_stack=config.n_stack)
    else:
        env = make_vec_env(config.env_id, n_envs=1, seed=config.seed)
    
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=config.n_eval_episodes) # type: ignore
    
    return mean_reward