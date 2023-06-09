import numpy as np
import gymnasium as gym
import random
import imageio
import os
from tqdm import tqdm
from pathlib import Path
import datetime
import json
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub import login
from huggingface_hub.repocard import metadata_eval_result, metadata_save


import pickle5 as pickle
#from tqdm.notebook import tqdm as tq

# Training parameters
n_training_episodes = 10000  # Total training episodes
learning_rate = 0.7  # Learning rate

# Evaluation parameters
n_eval_episodes = 100  # Total number of test episodes

# Environment parameters
env_id = "FrozenLake-v1"  # Name of the environment
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate
eval_seed = []  # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.0005  # Exponential decay rate for exploration prob

def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space,action_space))
    return Qtable

def greedy_policy(Qtable,state):
    action = np.argmax(Qtable[state][:])
    return action

def epsilon_greedy_policy(Qtable,state,epsilon):
    random_num = random.uniform(0,1)
    if random_num > epsilon:
        action = greedy_policy(Qtable,state)
    else:
        action = env.action_space.sample()
    return action

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in tqdm(range(n_training_episodes)):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable,state,epsilon)
            new_state,  reward, terminated, truncated, info = env.step(action)
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )
            if terminated or truncated:
                break
            state = new_state
    return Qtable

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()
        step = 0
        truncated = False
        terminated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

def record_video(env, Qtable, out_directory, fps=1):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []
    terminated = False
    truncated = False
    state, info = env.reset(seed=random.randint(0, 500))
    img = env.render()
    images.append(img)
    while not terminated or truncated:
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Qtable[state][:])
        state, reward, terminated, truncated, info = env.step(
            action
        )  # We directly put next_state = state for recording logic
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)

desc=["SFFF", "FHFH", "FFFH", "HFFG"]
env = gym.make('FrozenLake-v1', desc, is_slippery=False, render_mode="rgb_array").env


#print("_____OBSERVATION SPACE_____ \n")
#print("Observation Space", env.observation_space)
#print("Sample observation", env.observation_space.sample())

#print("\n _____ACTION SPACE_____ \n")
#print("Action Space Shape", env.action_space.n)
#print("Action Space Sample", env.action_space.sample())  # Take a random action

state_space = env.observation_space.n
print("There are ", state_space, " possible states")
action_space = env.action_space.n
print("There are ", action_space, " possible actions")

Qtable_frozenlake = initialize_q_table(state_space,action_space)

Qtable_frozenlake = train(n_training_episodes,min_epsilon,max_epsilon,decay_rate,env,max_steps,Qtable_frozenlake)

# Evaluate our Agent
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
video_path = "./replay.mp4" #fill this field
record_video(env, Qtable_frozenlake, video_path)