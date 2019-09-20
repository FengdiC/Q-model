
import numpy as np

class trajectory_dataset:
    def __init__(self, num_actions,max_trajectories=30, history_length=4):
        self.trajectories = []
        self.rewards = []
        self.num_actions = num_actions
        self.num_traj = 0
        self.max_trajectories = max_trajectories
        self.min_reward = -1
        self.history_length = history_length
        for i in range(max_trajectories):
            self.trajectories.append([])
            self.rewards.append([])

    def add_trajectory(self, trajectory_list, traj_reward):
        if traj_reward < self.min_reward:
            return
        if self.num_traj == self.max_trajectories:
            #
            index = -1
            min_reward = 50000000
            for i in range(self.max_trajectories):
                if self.rewards[i] < min_reward:
                    index = i
                    min_reward = self.rewards[i]
        else:
            index = self.num_traj
        self.rewards[index] = traj_reward
        self.trajectories[index] = trajectory_list

        for i in range(self.max_trajectories):
            if self.rewards[i] < self.min_reward:
                self.min_reward = self.rewards[i]
        self.num_traj += 1

    def get_minibatch(self, batch_size=32, img_dim=[84, 84]):
        #Needs to be in format 32, 84, 84, 4
        trajectories_indices = np.random.permutation(self.num_traj)
        states = np.zeros((batch_size, img_dim[0], img_dim[1], self.history_length))
        actions = np.zeros((batch_size, self.num_actions))
        terminal = np.zeros((batch_size,))
        rewards = np.zeros((batch_size,))
        weights = np.zeros((batch_size,))

        for i in range(batch_size):
            index = np.random.randint(self.history_length, len(self.trajectories[trajectories_indices[i]]))
            states[i] = self.trajectories[trajectories_indices[i]][index - self.history_length:index]
        return


