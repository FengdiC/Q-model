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


class toy_maze:
    def __init__(self, file,expert_dir='./',grid=10, final_reward=2, reward=1,cost=-0.01, obstacle_cost=0, level=15,expert=True,
                 agent_history_length=1):
        self.grid = grid
        self.expert_dir=expert_dir
        self.final_reward = final_reward
        self.reward = reward
        self.cost = cost/grid
        self.danger = -final_reward
        self.data = pickle.load(open(self.expert_dir+file, 'rb'))
        self.obstacles_cost = obstacle_cost
        self.level=0
        self.total_level=level
        self.n_actions=4
        self.agent_history_length = agent_history_length

        self.terminal=False
        self.reset()
        if expert:
            self.generate_expert_data()
            
    def restart(self):
        self.level=0

    def reset(self,eval=False):
        self.current_state_x = 0
        self.current_state_y = 0
        self.board = np.zeros((self.grid,self.grid,1))
        self.board[self.current_state_x,self.current_state_y]+=0.5
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

        #current_state
        #termination
        #reward
        #obstacle
        #danger zone
        new_representation = np.zeros((self.grid, self.grid, 5))
        for i in range(self.grid):
            for j in range(self.grid):
                if self.board[i, j] == 2:
                    new_representation[i, j, 2] = 1
                    new_representation[self.current_state_x, self.current_state_y, 1] = 1
                elif self.board[i, j] == 1:
                    new_representation[i, j, 2] = 1
                elif self.board[i, j] == -2:
                    new_representation[i, j, 4] = 1
                elif self.board[i, j] == -0.5:
                    new_representation[i, j, 3] = 1
        new_representation[self.current_state_x, self.current_state_y, 0] = 1
        self.state = new_representation


        # new_representation = np.zeros((self.grid, self.grid, 4))
        # mx0, my0 = self.current_state_x, max(0, self.current_state_y - 1)
        # mx1, my1 = self.current_state_x, min(self.grid - 1, self.current_state_y + 1)
        # mx2, my2 = max(0, self.current_state_x - 1), self.current_state_y
        # mx3, my3 = min(self.grid - 1, self.current_state_x + 1), self.current_state_y
        # new_representation[mx0, mx0, 2] = 1
        # new_representation[mx1, mx1, 2] = 1
        # new_representation[mx2, mx2, 2] = 1
        # new_representation[mx3, mx3, 2] = 1
        # new_representation[mx0, mx0, 0] = self.cost
        # new_representation[mx1, mx1, 0] = self.cost
        # new_representation[mx2, mx2, 0] = self.cost
        # new_representation[mx3, mx3, 0] = self.cost
        # for i in range(self.grid):
        #     for j in range(self.grid):
        #         if self.board[i, j] == 2:
        #             new_representation[i, j, 0] = 1
        #             new_representation[self.current_state_x, self.current_state_y, 1] = -1
        #         elif self.board[i, j] == 1:
        #             new_representation[i, j, 0] = 0.5
        #         elif self.board[i, j] == -2:
        #             new_representation[i, j, 0] = -1
        #         elif self.board[i, j] == -0.5:
        #             new_representation[i, j, 2] = -1
        #             new_representation[i, j, 0] = self.obstacles_cost
        #         new_representation[i, j, 3] = (np.abs(self.current_state_x - i) + np.abs(self.current_state_y - j))/self.grid - 1
        # new_representation[self.current_state_x, self.current_state_y, 1] = 1
        # self.state = new_representation
        # self.state = np.repeat(self.board/2.0,self.agent_history_length,axis=2)
        # print(self.state.shape)
        return self.state

    def step(self, action):
        x=self.current_state_x
        y =self.current_state_y
        self.board[self.current_state_x,self.current_state_y] = 0

        past_x = x
        past_y = y

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
                reward = -self.cost
        # if blocked by obstacles
        if self.board[x,y]==-0.5 or (past_x == x and past_y == y):
            reward = self.obstacles_cost
        else:
            self.current_state_x = x
            self.current_state_y = y

        self.timestep += 1
        self.terminal = terminal
        self.board[self.current_state_x,self.current_state_y] = 0.5

        # new_representation = np.zeros((self.grid, self.grid, 4))
        # mx0, my0 = self.current_state_x, max(0, self.current_state_y - 1)
        # mx1, my1 = self.current_state_x, min(self.grid - 1, self.current_state_y + 1)
        # mx2, my2 = max(0, self.current_state_x - 1), self.current_state_y
        # mx3, my3 = min(self.grid - 1, self.current_state_x + 1), self.current_state_y
        # new_representation[mx0, mx0, 2] = 1
        # new_representation[mx1, mx1, 2] = 1
        # new_representation[mx2, mx2, 2] = 1
        # new_representation[mx3, mx3, 2] = 1
        # new_representation[mx0, mx0, 0] = self.cost
        # new_representation[mx1, mx1, 0] = self.cost
        # new_representation[mx2, mx2, 0] = self.cost
        # new_representation[mx3, mx3, 0] = self.cost
        # for i in range(self.grid):
        #     for j in range(self.grid):
        #         if self.board[i, j] == 2:
        #             new_representation[i, j, 0] = 1
        #             new_representation[self.current_state_x, self.current_state_y, 1] = -1
        #         elif self.board[i, j] == 1:
        #             new_representation[i, j, 0] = 0.5
        #         elif self.board[i, j] == -2:
        #             new_representation[i, j, 0] = -1
        #         elif self.board[i, j] == -0.5:
        #             new_representation[i, j, 2] = -1
        #             new_representation[i, j, 0] = self.obstacles_cost
        #         new_representation[i, j, 3] = (np.abs(self.current_state_x - i) + np.abs(self.current_state_y - j))/self.grid - 1
        # new_representation[self.current_state_x, self.current_state_y, 1] = 1
        # self.state = new_representation

        return self.state, reward, terminal

    def generate_expert_data(self, min_expert_frames=5500):
        print("Creating Expert Data ... ")
        data = pickle.load(open(self.expert_dir+'long_maze_1.pkl', 'rb'))
        expert_action=data['actions']
        expert = {}
        num_batches = math.ceil(min_expert_frames / len(expert_action))
        num_expert = num_batches * len(expert_action)

        expert_frames = np.zeros((num_expert, 10,10, 5), np.float32)
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
        with open(self.expert_dir+'expert_maze', 'wb') as fout:
            pickle.dump(expert, fout)

    def print_state(self):
        print(self.current_state_x, self.current_state_y)

    def get_default_shape(self):
        return [self.grid, self.grid]

def play():
    key_to_action = {'a':0,'d':1,'w':2,'s':3}
    env= toy_maze('mazes',expert=False)
    pressed_keys = []
    running = True
    env_done = False
    env.restart()
    obs = env.reset(eval=True)
    action=0

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
            obs = env.reset(eval=False)
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
            pickle.dump(current_data, open("long_maze_" + str(num_traj) + ".pkl", "wb"), protocol=4)
            data_list = []
        else:
            obs_next, rew, terminal = env.step(action)
            data_list.append([action, obs, rew, terminal])
            obs=obs_next
            count += 1
            if env.terminal and env.level>10:
                env_done= True
            if env.terminal:
                env.reset(eval=True)

        if obs is not None:
            plot_state(obs[:,:,-1])
            key = input("action: \n")
            if 'a' in key:
                action = key_to_action['a']
            elif 'w' in key:
                action = key_to_action['w']
            elif 'd' in key:
                action = key_to_action['d']
            else:
                action = key_to_action['s']


def plot_state(state,grid=10):
    square = state[:,:]*200+200
    fig, ax = plt.subplots()
    im = ax.imshow(square)

    for i in range(grid):
        for j in range(grid):
            text = ax.text(j, i, square[i, j],
                           ha="center", va="center", color="w")
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()

def generate_maze(grid):
    count = 0
    end_state = [np.random.randint(0, grid ), np.random.randint(0, grid )]
    while end_state[0] == 0 and end_state[1] == 0:
        end_state = [np.random.randint(0, grid ), np.random.randint(0, grid )]
    obstacles=[]
    rewards=[]
    dangers = []
    while count<8:
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
    while count < 16:
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
    for i in range(99):
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

# maze=pickle.load(open('mazes','rb'))
# for i in range(85):
#     e, d, r, o = generate_maze(10)
#     print("rewards: ", r)
#     print("end_state: ", e)
#     print("dangers: ", d)
#     print("obstables: ", o)
#     maze['end_state'].insert(0,e)
#     maze['rewards'].insert(0,r)
#     maze['obstacles'].insert(0,o)
#     maze['dangers'].insert(0,d)
# with open('mazes_large', 'wb') as fout:
#     pickle.dump(maze, fout)