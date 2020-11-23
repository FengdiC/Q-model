import tensorflow as tf
import numpy as np
import os
import time
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')
import utils
import PriorityBuffer
import pickle
from tensorboard_logger import tensorflowboard_logger
import math


class toy_maze:
    def __init__(self, grid, final_reward=1, min_expert_frames=512, expert=True):
        self.grid = grid
        self.final_reward = final_reward
        self.reward = reward
        self.cost = -0.01/grid
        self.danger = -final_reward
        self.end_state = end_state
        self.obstacles = obstacles
        self.dangers=dangers
        self.rewards = rewards
        self.reset()
        if expert:
            self.generate_expert_data(min_expert_frames=min_expert_frames)

    def reset(self):
        self.current_state_x = 0
        self.current_state_y = 0
        self.timestep = 0
        return np.array([1,1])

    def get_current_state(self):
        result = np.zeros((self.grid, self.grid), dtype=np.uint8)
        result[self.current_state_x, self.current_state_y] = 1
        result = np.reshape(result, [self.grid * self.grid])
        return result

    def step(self, action):
        assert not (action != 0 and action != 1), "invalid action"
        x=self.current_state_x
        y =self.current_state_y
        if action == 0: #left
            x=self.current_state_x - 1
        elif action ==1: #right
            x= self.current_state_x + 1
        elif action==2: #up
            y = self.current_state_y + 1
        else: #down
            y = self.current_state_y - 1
        x = np.clip(x, 0, self.grid - 1)
        y = np.clip(y, 0, self.grid - 1)
        dist_to_e = np.abs(x - self.end_state[0]) + np.abs(y - self.end_state[1])
        if dist_to_e == 0:
            terminal = 1
            reward = self.final_reward
            print("Reach final reward")
        else:
            terminal = 0
            dist_to_d = min([np.abs(x - z[0]) + np.abs(y - z[1]) for z in self.dangers] or [99])
            dist_to_r = min([np.abs(x - z[0]) + np.abs(y - z[1]) for z in self.rewards] or [99])
            if dist_to_d == 0:
                reward=self.danger
            elif dist_to_r == 0:
                reward =self.reward
            else:
                reward = self.cost
        dist_to_o = min([np.abs(x - z[0]) + np.abs(y - z[1]) for z in self.obstacles] or [99])
        if dist_to_o == 0:
            reward =0
        else:
            self.current_state_x = x
            self.current_state_y = y

        self.timestep += 1
        return np.array([self.current_state_x+1, self.current_state_y+1]), reward, terminal

    def generate_expert_data(self, min_expert_frames=512, expert_ratio=1):
        print("Creating Expert Data ... ")
        expert = {}
        half_expert_traj = self.grid // expert_ratio
        num_batches = math.ceil(min_expert_frames / half_expert_traj)
        num_expert = num_batches * half_expert_traj

        expert_frames = np.zeros((num_expert, 2), np.uint8)
        rewards = np.zeros((num_expert,), dtype=np.float32)
        terminals = np.zeros((num_expert,), np.uint8)

        current_index = 0
        for i in range(num_batches):
            current_state = self.reset()
            for j in range(half_expert_traj):
                s, r, t = self.step(1)
                expert_frames[current_index] = current_state
                current_state = s
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

def generate_maze(grid):
    count = 0
    end_state = [np.random.randint(0, grid ), np.random.randint(0, grid )]
    while end_state[0] == 0 and end_state[1] == 0:
        end_state = [np.random.randint(0, grid ), np.random.randint(0, grid )]
    obstacles=[]
    rewards=[]
    dangers = []
    while count<2:
        state = [np.random.randint(0,grid),np.random.randint(0,grid)]
        if state[0]==0 and state[1]==0:
            continue
        dist_to_e = np.abs(state[0] - end_state[0]) + np.abs(state[1] - end_state[1])
        if dist_to_e == 0:
            continue
        dist_to_d = min([np.abs(state[0] - x[0]) + np.abs(state[1] - x[1]) for x in dangers] or [99])
        if dist_to_d==0:
            continue
        dangers.append(state)
        count+=1
    count = 0
    while count < 4:
        state = [np.random.randint(0, grid ), np.random.randint(0, grid )]
        if state[0]==0 and state[1]==0:
            continue
        dist_to_e = np.abs(state[0] - end_state[0]) + np.abs(state[1] - end_state[1])
        if dist_to_e == 0:
            continue
        dist_to_d = min([np.abs(state[0] - x[0]) + np.abs(state[1] - x[1]) for x in dangers]or [99])
        if dist_to_d == 0:
            continue
        dist_to_r = min([np.abs(state[0] - x[0]) + np.abs(state[1] - x[1]) for x in rewards]or [99])
        if dist_to_r ==0:
            continue
        rewards.append(state)
        count+=1
    count=0
    while count<min(grid-1,10):
        state = [np.random.randint(0,grid),np.random.randint(0,grid)]
        if state[0]==0 and state[1]==0:
            continue
        dist_to_e = np.abs(state[0] - end_state[0]) + np.abs(state[1] - end_state[1])
        if dist_to_e == 0:
            continue
        dist_to_d = min([np.abs(state[0] - x[0]) + np.abs(state[1] - x[1]) for x in dangers] or [99])
        if dist_to_d == 0:
            continue
        dist_to_r = min([np.abs(state[0] - x[0]) + np.abs(state[1] - x[1]) for x in rewards] or [99])
        if dist_to_r == 0:
            continue
        dist_to_o = min([np.abs(state[0]-x[0])+np.abs(state[1]-x[1])+
                         np.abs(np.abs(state[0]-x[0])-np.abs(state[1]-x[1])) for x in obstacles]or [99])
        if dist_to_o <=2:
            continue
        obstacles.append(state)
        count+=1
    print("rewards: ",rewards)
    print("end_state: ", end_state)
    print("dangers: ",dangers)
    print("obstables: ",obstacles)
    return end_state,dangers,rewards,obstacles

generate_maze(6)