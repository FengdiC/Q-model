import tensorflow as tf
import numpy as np
import os
import time
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')
import utils
import PriorityBuffer
import pickle
import copy
import math
import matplotlib.pyplot as plt

expert_action = [1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,0,0,0,0,3,0,0,2,2,2,1,1,1,1,2,2,0,0,0,2,3,0,0,3,3,
                 3,1,1,1,1,1,1,1,1,3,3,3,3,3,3,0,0,2,2,2,2,2,0,0,0,0]

class toy_maze:
    def __init__(self, file,grid=10, final_reward=2, reward=1,cost=-0.01,expert=False):
        self.grid = grid
        self.final_reward = final_reward
        self.reward = reward
        self.cost = cost/grid
        self.danger = -final_reward
        self.data = pickle.load(open(file, 'rb'))
        self.level=1
        self.n_actions=4

        self.terminal=False
        self.reset()
        if expert:
            self.generate_expert_data()

    def reset(self):
        self.current_state_x = 0
        self.current_state_y = 0
        self.timestep = 0

        self.level=np.random.randint(1,8)

        self.end_state = copy.deepcopy(self.data['end_state'][-self.level])
        self.obstacles = copy.deepcopy(self.data['obstacles'][-self.level][:self.grid-1])
        self.dangers = copy.deepcopy(self.data['dangers'][-self.level])
        self.rewards = copy.deepcopy(self.data['rewards'][-self.level])

        self.terminal = False
        state = [0,0]
        for x in self.rewards:
            state += x
        state += self.end_state
        for x in self.obstacles:
            state += x
        for x in self.dangers:
            state+=x
        return np.array(state)/(0.5*self.grid)-1

    def get_current_state(self):
        result = np.zeros((self.grid, self.grid), dtype=np.uint8)
        result[self.current_state_x, self.current_state_y] = 1
        result = np.reshape(result, [self.grid * self.grid])
        return result

    def step(self, action):
        x=self.current_state_x
        y =self.current_state_y

        #take actions
        if action == 0: #left
            y=self.current_state_y - 1
        elif action ==1: #right
            y= self.current_state_y + 1
        elif action==2: #up
            x = self.current_state_x - 1
        else: #down
            x = self.current_state_x + 1
        x = np.clip(x, 0, self.grid - 1)
        y = np.clip(y, 0, self.grid - 1)

        # measure terminal and rewards
        dist_to_e = np.abs(x - self.end_state[0]) + np.abs(y - self.end_state[1])
        if dist_to_e == 0:
            terminal = 1
            reward = self.final_reward
            print("Reach final reward")
        else:
            terminal = 0
            dist_to_d = min([np.abs(x - z[0]) + np.abs(y - z[1]) for z in self.dangers] or [99])
            if dist_to_d == 0:
                reward=self.danger
                print("Reach Danger")
            else:
                reward = self.cost
            for z in self.rewards:
                if np.abs(x - z[0]) + np.abs(y - z[1])==0:
                    reward = self.reward
                    z[0]=-1
                    z[1]=-1

        # if blocked by obstacles
        dist_to_o = min([np.abs(x - z[0]) + np.abs(y - z[1]) for z in self.obstacles] or [99])
        if dist_to_o == 0:
            reward =0
        else:
            self.current_state_x = x
            self.current_state_y = y

        # generate state
        state = [self.current_state_x,self.current_state_y]
        for x in self.rewards:
            state += x
        state += self.end_state
        for x in self.obstacles:
            state += x
        for x in self.dangers:
            state += x

        self.timestep += 1
        self.terminal = terminal
        return np.array(state)/(0.5*self.grid)-1, reward, terminal

    def generate_expert_data(self, min_expert_frames=1024):
        print("Creating Expert Data ... ")
        data = pickle.load(open('/home/yutonyan/Q-model/full_maze_2.pkl', 'rb'))
        expert_action = data['actions']
        expert = {}
        num_batches = math.ceil(min_expert_frames / len(expert_action))
        num_expert = num_batches * len(expert_action)

        expert_frames = np.zeros((num_expert, 100), np.float32)
        rewards = np.zeros((num_expert,), dtype=np.float32)
        actions = np.zeros((num_expert,), dtype=np.float32)
        terminals = np.zeros((num_expert,), np.uint8)

        current_index = 0
        for i in range(num_batches):
            current_state = self.reset()
            self.level=1
            for j in range(len(expert_action)):
                action = expert_action[j]
                expert_frames[current_index] = current_state
                s, r, t = self.step(action)
                current_state=s
                rewards[current_index] = r
                actions[current_index] = action
                terminals[current_index] = t
                current_index += 1
                if t:
                    current_state = self.reset()
                    self.level=2
        expert['actions'] = actions
        expert['frames'] = expert_frames
        expert['reward'] = rewards
        expert['terminal'] = terminals
        with open('/home/yutonyan/Q-model/expert_maze', 'wb') as fout:
            pickle.dump(expert, fout)

    def print_state(self):
        print(self.current_state_x, self.current_state_y)

    def get_default_shape(self):
        return [self.grid, self.grid]
        
class toy_maze_grid:
    def __init__(self, file,grid=10, final_reward=2, reward=1,cost=-0.01,level=6,expert=True):
        self.grid = grid
        self.final_reward = final_reward
        self.reward = reward
        self.cost = cost/grid
        self.danger = -final_reward
        self.data = pickle.load(open(file, 'rb'))
        self.level=0
        self.total_level=level
        self.n_actions=4

        self.terminal=False
        self.reset()
        if expert:
            self.generate_expert_data()
            
    def restart(self):
        self.level=0

    def reset(self,eval=False):
        self.current_state_x = 0
        self.current_state_y = 0
        self.board = np.zeros((self.grid,self.grid))
        self.board[self.current_state_x,self.current_state_y]=0.5
        self.timestep = 0

        if eval:
            if self.level<self.total_level:
                self.level+=1
            else:
                self.level=1
        else:
            self.level=np.random.randint(1,self.total_level+1)

        self.end_state = copy.deepcopy(self.data['end_state'][-self.level])
        self.obstacles = copy.deepcopy(self.data['obstacles'][-self.level][:self.grid-1])
        self.dangers = copy.deepcopy(self.data['dangers'][-self.level])
        self.rewards = copy.deepcopy(self.data['rewards'][-self.level])

        self.terminal = False
        for x in self.rewards:
            self.board[int(x[0]),int(x[1])]=1
        self.board[int(self.end_state[0]),int(self.end_state[1])]=2
        for x in self.obstacles:
            self.board[int(x[0]),int(x[1])]=-0.5
        for x in self.dangers:
            self.board[int(x[0]),int(x[1])]=-2
        return self.board.flatten()/2.0

    def get_current_state(self):
        result = np.zeros((self.grid, self.grid), dtype=np.uint8)
        result[self.current_state_x, self.current_state_y] = 1
        result = np.reshape(result, [self.grid * self.grid])
        return result

    def step(self, action):
        x=self.current_state_x
        y =self.current_state_y
        self.board[self.current_state_x,self.current_state_y]=0

        #take actions
        if action == 0: #left
            y=self.current_state_y - 1
        elif action ==1: #right
            y= self.current_state_y + 1
        elif action==2: #up
            x = self.current_state_x - 1
        else: #down
            x = self.current_state_x + 1
        x = np.clip(x, 0, self.grid - 1)
        y = np.clip(y, 0, self.grid - 1)
        
        if self.board[x,y]==2:
            terminal = 1
            reward = self.final_reward
            # print("Reach Final Reward")
        else:
            terminal =0
            if self.board[x,y]==-2:
                reward=self.danger
                # print("Reach Danger")
            elif self.board[x,y]==1:
                reward = self.reward
                self.board[x,y]=0
            else:
                reward = self.cost

        # if blocked by obstacles
        if self.board[x,y]==0.5:
            reward =0
        else:
            self.current_state_x = x
            self.current_state_y = y

        # generate state
        state = [self.current_state_x,self.current_state_y]
        for x in self.rewards:
            state += x
        state += self.end_state
        for x in self.obstacles:
            state += x
        for x in self.dangers:
            state += x

        self.timestep += 1
        self.terminal = terminal
        self.board[self.current_state_x,self.current_state_y]=0.5
        return self.board.flatten()/2.0, reward, terminal

    def generate_expert_data(self, min_expert_frames=1500):
        print("Creating Expert Data ... ")
        data = pickle.load(open('/home/yutonyan/Q-model/full_maze_2.pkl', 'rb'))
        expert_action=data['actions']
        expert = {}
        num_batches = math.ceil(min_expert_frames / len(expert_action))
        num_expert = num_batches * len(expert_action)

        expert_frames = np.zeros((num_expert, 100), np.float32)
        rewards = np.zeros((num_expert,), dtype=np.float32)
        actions = np.zeros((num_expert,), dtype=np.float32)
        terminals = np.zeros((num_expert,), np.uint8)

        current_index = 0
        for i in range(num_batches):
            self.level =0 
            current_state = self.reset(eval=True)
            for j in range(len(expert_action)):
                action = expert_action[j]
                expert_frames[current_index] = current_state
                s, r, t = self.step(action)
                current_state=s
                rewards[current_index] = r
                actions[current_index] = action
                terminals[current_index] = t
                current_index += 1
                if t:
                    current_state = self.reset(eval=True)
        expert['actions'] = actions
        expert['frames'] = expert_frames
        expert['reward'] = rewards
        expert['terminal'] = terminals
        with open('/home/yutonyan/Q-model/expert_maze', 'wb') as fout:
            pickle.dump(expert, fout)

    def print_state(self):
        print(self.current_state_x, self.current_state_y)

    def get_default_shape(self):
        return [self.grid, self.grid]

def play():
    key_to_action = {'a':0,'d':1,'w':2,'s':3}
    env= toy_maze('mazes')
    pressed_keys = []
    running = True
    env_done = True

    num_traj = 0
    count =0
    data_list = []

    current_data = {}
    current_data["frames"] = []
    current_data["reward"] = []
    current_data["actions"] = []
    current_data["terminal"] = []

    while running:
        if env_done:
            env_done = False
            num_traj += 1
            obs = env.reset()
            print(num_traj, count)

            for i in range(len(data_list)):
                action = data_list[i][0]
                obs = data_list[i][1]
                rew = data_list[i][2]
                terminal = data_list[i][3]
                current_data["frames"].append(obs)
                current_data["reward"].append(rew)
                current_data["actions"].append(action)
                current_data["terminal"].append(terminal)
                # replay_mem.add(obs[:, :, 0], action, rew, terminal)
            pickle.dump(current_data, open("full_maze_" + str(num_traj) + ".pkl", "wb"), protocol=4)
            data_list = []
        else:
            obs, rew, terminal = env.step(action)
            data_list.append([action, obs, rew, terminal])
            count += 1
            if env.terminal and env.level>3:
                env_done= True
            if env.terminal:
                env.reset()

        if obs is not None:
            plot_state(obs)
            key = input("action: \n")
            action = key_to_action[key]


def plot_state(state,grid=10):
    state = state.tolist()
    square = np.zeros((grid,grid))
    square[int(state.pop(0)-1),int(state.pop(0)-1)] = 1000
    for i in range(4):
        square[int(state.pop(0) - 1), int(state.pop(0) - 1)] = 600
    square[int(state.pop(0) - 1), int(state.pop(0) - 1)] =900
    for i in range(grid-1):
        square[int(state.pop(0) - 1), int(state.pop(0) - 1)] = 200
    for i in range(2):
        square[int(state.pop(0) - 1), int(state.pop(0) - 1)] = 100
    fig, ax = plt.subplots()
    im = ax.imshow(square)

    for i in range(grid):
        for j in range(grid):
            text = ax.text(j, i, square[i, j],
                           ha="center", va="center", color="w")
    fig.tight_layout()
    plt.show()

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
        dist_to_d = min([np.abs(state[0] - x[0]) + np.abs(state[1] - x[1]) +
                             np.abs(np.abs(state[0] - x[0]) - np.abs(state[1] - x[1])) for x in dangers] or [99])
        if dist_to_d <=2:
            continue
        dist_to_r = min([np.abs(state[0] - x[0]) + np.abs(state[1] - x[1]) for x in rewards] or [99])
        if dist_to_r == 0:
            continue
        dist_to_o = min([np.abs(state[0]-x[0])+np.abs(state[1]-x[1])+
                         np.abs(np.abs(state[0]-x[0])-np.abs(state[1]-x[1])) for x in obstacles]or [99])
        if dist_to_o <=2:
            continue

        length = min(state[0],state[1],grid-1-state[0],grid-1-state[1])
        l=0; condition=True
        action = np.random.randint(0,4)
        while l <=min(length,grid//3) and condition:
            obstacles.append(state.copy())
            count+=1
            l+=1
            if action == 0:  # left
                state[0] = state[0] - 1
            elif action == 1:  # right
                state[0] = state[0] + 1
            elif action == 2:  # up
                state[1] = state[1] + 1
            else:  # down
                state[1] = state[1] -1
            if state[0] == 0 and state[1] == 0:
                condition=False
            dist_to_e = np.abs(state[0] - end_state[0]) + np.abs(state[1] - end_state[1])
            if dist_to_e == 0:
                condition=False
            dist_to_d = min([np.abs(state[0] - x[0]) + np.abs(state[1] - x[1]) +
                             np.abs(np.abs(state[0] - x[0]) - np.abs(state[1] - x[1])) for x in dangers] or [99])
            if dist_to_d<=2:
                condition=False
            dist_to_r = min([np.abs(state[0] - x[0]) + np.abs(state[1] - x[1]) for x in rewards] or [99])
            if dist_to_r == 0:
                condition=False
            dist_to_o = min([np.abs(state[0] - x[0]) + np.abs(state[1] - x[1]) +
                             np.abs(np.abs(state[0] - x[0]) - np.abs(state[1] - x[1])) for x in obstacles[:-(l)]] or [99])
            if dist_to_o <= 2:
                condition=False
    # print("rewards: ",rewards)
    # print("end_state: ", end_state)
    # print("dangers: ",dangers)
    # print("obstables: ",obstacles)
    return end_state,dangers,rewards,obstacles

def mazes_generation():
    maze={'end_state':[],'rewards':[],'dangers':[],'obstacles':[]}
    e,d,r,o = generate_maze(10)
    print("rewards: ", r)
    print("end_state: ", e)
    print("dangers: ",d)
    print("obstables: ",o)
    maze['end_state'].append(e)
    maze['rewards'].append(r)
    maze['obstacles'].append(o)
    maze['dangers'].append(d)
    for i in range(6):
        e, d, r, o = generate_maze(10)
        print("rewards: ", r)
        print("end_state: ", e)
        print("dangers: ", d)
        print("obstables: ", o)
        maze['end_state'].append(e)
        maze['rewards'].append(r)
        maze['obstacles'].append(o)
        maze['dangers'].append(d)
    with open('test_mazes_2', 'wb') as fout:
        pickle.dump(maze, fout)

# play()
# mazes_generation()
