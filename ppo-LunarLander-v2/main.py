from stable_baselines3 import ppo
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import login

env_id = "LunarLander-v2"
model_architecture = "PPO"

env = gym.make(env_id)


model = ppo.PPO(
    "MlpPolicy",
    env, 
    n_epochs= 4,
    gamma=0.999,
    n_steps= 1024,
    ent_coef=0.1,
    gae_lambda=0.97,
    verbose=1
)

model.learn(total_timesteps=(1000000))
"""
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample())

print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample())
"""

eval_env = Monitor(gym.make("LunarLander-v2"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")