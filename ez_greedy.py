import tensorflow as tf
import numpy as np
import os
import time
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')
import utils
import PriorityBuffer
import pickle
import math

class toy_env:
    def __init__(self, grid, final_reward=1, min_expert_frames=512):
        self.grid = grid
        self.final_reward = final_reward
        self.reset()
        self.generate_expert_data(min_expert_frames=min_expert_frames)

    def reset(self):
        self.current_state_x = 0
        self.current_state_y = 0
        self.timestep = 0
        return self.get_current_state()

    def get_current_state(self):
        result = np.zeros((self.grid, self.grid), dtype=np.uint8)
        result[self.current_state_x, self.current_state_y] = 1
        result = np.reshape(result, [self.grid * self.grid])
        return result

    def step(self, action):
        assert not (action != 0 and action != 1), "invalid action"
        if action == 0:
            self.current_state_x -= 1
            self.current_state_y += 1
            reward = 0
        else:
            self.current_state_x += 1
            self.current_state_y += 1
            reward = -0.01/self.grid
        
        self.current_state_x = np.clip(self.current_state_x, 0, self.grid - 1)
        self.current_state_y = np.clip(self.current_state_y, 0, self.grid - 1)

        if (self.current_state_x == self.grid - 1) and (self.current_state_y == self.grid - 1):
            reward = self.final_reward
        self.timestep += 1
        if self.timestep >= self.grid - 1:
            terminal = 1
        else:
            terminal = 0
        if self.current_state_x == self.grid - 1:
            print(terminal, self.current_state_x, self.current_state_y, reward)
        return self.get_current_state(), reward, terminal

    def generate_expert_data(self, min_expert_frames=512, expert_ratio=2):
        expert = {}
        half_expert_traj = min(self.grid//expert_ratio,20)
        print("Expert Generated!")
        print("Expert Length: ", half_expert_traj, "out of", self.grid)
        num_batches = math.ceil(min_expert_frames/half_expert_traj)
        num_expert = num_batches * half_expert_traj
        
        expert_frames = np.zeros((num_expert, self.grid * self.grid), np.uint8)
        rewards = np.zeros((num_expert, ), dtype=np.float32)
        terminals = np.zeros((num_expert,), np.uint8)

        current_index = 0
        for i in range(num_batches):
            current_state = self.reset()
            for j in range(half_expert_traj):
                s, r, t = self.step(1)
                expert_frames[current_index] = s 
                rewards[current_index] = r 
                terminals[current_index] = t
                current_index += 1
        expert['actions'] = np.ones((num_expert,), dtype=np.uint8)
        expert['frames'] = expert_frames
        expert['reward'] = rewards
        expert['terminal'] = terminals
        with open('expert_toy', 'wb') as fout:
            pickle.dump(expert, fout)

    def print_state(self):
        print(self.current_state_x, self.current_state_y)

    def get_default_shape(self):
        return [self.grid, self.grid]


def ez_action_sampling(mu, n_actions, past_action, duration_left):
    #sample randomly an action ..
    if duration_left <= 0:
        past_action = np.random.randint(0, n_actions)
        duration_left = np.random.zipf(mu)
    duration_left -= 1
    return past_action, duration_left


def run_trajectory(env, frame_num, eps_length, grid, mu, n_actions): 
    episode_reward_sum = 0
    episode_length = 0

    duration_left = 0
    past_action = -1
    update_count = np.zeros((grid, grid, 2))
    for _ in range(eps_length):
        past_action, duration_left = ez_action_sampling(mu, n_actions, past_action, duration_left)
        next_frame, reward, terminal = env.step(past_action)
        update_count[next_frame[0], next_frame[1], past_action] += 1

        episode_length += 1
        episode_reward_sum += reward
        frame_num += 1
        if terminal:
            env.reset()
            break
    return episode_reward_sum, episode_length


def train(seed=0, mu=2):
    args = utils.argsparser()
    grid = args.grid_size
    MAX_EPISODE_LENGTH = grid - 1
    MAX_FRAMES = 50000000  # Total number of frames the agent sees
    n_actions = 2
    frame_number = 0
    initial_time = time.time()
    eps_rewards = []
    eps_number = 0
    env = toy_env(grid)
    while frame_number < MAX_FRAMES:
        eps_reward, eps_length = run_trajectory(env, frame_number, MAX_EPISODE_LENGTH, grid, mu, n_actions)
        frame_number += eps_length
        eps_number += 1
        if eps_reward > 0:
            print(frame_number, eps_reward, args.seed)
            exit()

train()