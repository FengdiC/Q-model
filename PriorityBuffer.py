import numpy as np
import random
import pickle
import operator


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)

class ReplayBuffer(object):
    def __init__(self, size, var =1.0, frame_height=84, frame_width=84,
                 agent_history_length=4, batch_size=32):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        print("Called Not Priority Queue!")
        self._maxsize = size
        self._next_idx = 0
        self.expert_idx = 0
        self.var = var

        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.max_reward = 1
        self.min_reward = 0
        self.count = 0

        # Pre-allocate memory
        self.actions = np.empty(self._maxsize, dtype=np.int32)
        self.diffs = np.empty(self._maxsize, dtype=np.float32)
        self.rewards = np.empty(self._maxsize, dtype=np.float32)
        self.frames = np.empty((self._maxsize, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self._maxsize, dtype=np.uint8)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.zeros((self.batch_size, self.agent_history_length,self.frame_height, self.frame_width,
                                ), dtype=np.float32)
        self.new_states = np.zeros((self.batch_size, self.agent_history_length,self.frame_height, self.frame_width,
                                    ), dtype=np.float32)
        self.indices = np.zeros(self.batch_size, dtype=np.int32)


    def add_expert(self, obs_t, reward, action,diff, done):
        reward = np.sign(reward) * np.log(1+np.abs(reward))
        self.actions[self.expert_idx] = action
        self.diffs[self.expert_idx] = diff
        self.frames[self.expert_idx] = np.squeeze(obs_t)
        self.rewards[self.expert_idx] = reward
        self.terminal_flags[self.expert_idx] = done

        self.count = min(self._maxsize, self.count + 1)
        self.expert_idx = (self.expert_idx + 1)
        if self.expert_idx > self._maxsize:
            print("Dataset too small, cannot add more expert actions .... ")
            quit()
        self._next_idx = max(self._next_idx, self.expert_idx)

    def __len__(self):
        return self.count

    def add(self, obs_t, reward, action, done):
        reward = np.sign(reward) * np.log(1+np.abs(reward))
        self.actions[self._next_idx] = action
        self.diffs[self._next_idx] = 0
        self.frames[self._next_idx] = obs_t
        self.rewards[self._next_idx] = reward
        self.terminal_flags[self._next_idx] = done

        self.count = min(self._maxsize, self.count + 1)
        self._next_idx = (self._next_idx + 1) % (self._maxsize)
        if self._next_idx < self.expert_idx:
            self._next_idx = self.expert_idx
    # def _encode_sample(self, idxes):
    #     obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
    #     for i in idxes:
    #         data = self._storage[i]
    #         obs_t, action, reward, obs_tp1, done = data
    #         obses_t.append(np.array(obs_t, copy=False))
    #         actions.append(np.array(action, copy=False))
    #         rewards.append(reward)
    #         obses_tp1.append(np.array(obs_tp1, copy=False))
    #         dones.append(done)
    #     return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index - self.agent_history_length + 1:index + 1]

    def _get_indices(self, batch_size):
        idxes = []
        while len(idxes) < batch_size:
            index = np.random.randint(self.agent_history_length + 1, self.count - 1)
            if index >= self.expert_idx and index - self.agent_history_length < self.expert_idx:
                continue
            if index >= self._next_idx and index - self.agent_history_length < self._next_idx:
                continue
            if np.sum(self.terminal_flags[index - self.agent_history_length - 1:index - 1]) > 0:
                continue
            idxes.append(index)

        # frame_indices = np.zeros((batch_size, self.agent_history_length + 1), dtype=np.uint32)
        # for i in range(len(idxes)):
        #     low_cap = idxes[i] - self.agent_history_length - 1
        #     high_cap = idxes[i]
        #     if index >= self.expert_idx and index - self.agent_history_length - 1 < self.expert_idx:
        #         low_cap = self.expert_idx
        #     if index >= self._next_idx and index - self.agent_history_length - 1 < self._next_idx:
        #         high_cap = self._next_idx
        #     if np.sum(self.terminal_flags[index - self.agent_history_length - 1:index - 1]) > 0:
        #         high_cap = self._next_idx
        #
        #     for j in range(self.agent_history_length + 1):
        #         frame_indices[i, j] = min(high_cap, max(low_cap, idxes[i] - self.agent_history_length - 1 + j))
        # return frame_indices
        return idxes

    def _get_expert_indices(self, batch_size):
        if self.expert_idx == 0:
            print("No expert data ...")
            quit()

        idxes = []
        while len(idxes) < batch_size:
            index = np.random.randint(self.agent_history_length + 1, self.expert_idx)
            if index >= 0 and index - self.agent_history_length < self.expert_idx:
                continue
            if index >= 0 and index - self.agent_history_length < self._next_idx:
                continue
            if np.sum(self.terminal_flags[index - self.agent_history_length:index]) > 0:
                 continue
            idxes.append(index)
        return idxes

    def load_expert_data(self, path):
        data = pickle.load(open(path, 'rb'))
        num_data = len(data['frames'])
        print("load expert: ",num_data)
        print("Loading Expert Data ... ")
        for i in range(num_data):
            self.add_expert(obs_t=data['frames'][i], reward=data['reward'][i], action=data['actions'][i],
                     diff = self.var, done=data['terminal'][i])
            #print(data['reward'][i], np.sum(data['terminal']))

        #Reward check
        print("Min Reward: ", np.min(self.rewards[self.rewards > 0]), "Max Reward: ", np.max(self.rewards[self.rewards > 0]))
        print(self.count, "Expert Data loaded ... ")



    def sample(self, batch_size, expert=False):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if expert:
            idxes = self._get_expert_indices(batch_size)
        else:
            idxes = self._get_indices(batch_size)


        idxes = np.array(idxes)
        selected_actions = self.actions[idxes]
        selected_diffs = self.diffs[idxes]
        selected_rewards =self.rewards[idxes]
        selected_terminal = self.terminal_flags[idxes]
        for i, idx in enumerate(idxes):
            
            #if np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
            #    terminal_idx = idx - self.agent_history_length + np.argmax(self.terminal_flags[idx - self.agent_history_length:idx])
            #    selected_actions[i] = self.actions[terminal_idx]
            #    selected_rewards[i] = self.rewards[terminal_idx]
            #    selected_terminal[i] = self.terminal_flags[terminal_idx]
            #    self.states[i] = self._get_state(terminal_idx)
            #    self.new_states[i] = self._get_state(terminal_idx)
            #else:
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
        # for i, idx in enumerate(idxes):
        #     self.states[i] =  self._get_state(idx - 1)
        #     self.new_states[i] = self._get_state(idx)
        # self.states = (self.states - 127.5)/127.5
        # self.new_states = (self.new_states - 127.5)/127.5
        return np.transpose(self.states, axes=(0, 2, 3, 1)), selected_actions,selected_diffs, \
               selected_rewards, np.transpose(self.new_states, axes=(0, 2, 3, 1)), selected_terminal


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha,var):
        print("Priority Queue!")
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size,var)
        assert alpha >= 0
        self._alpha = alpha
        self.var= var

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 5

        #self.expert_indices = {}

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def add_expert(self, *args, **kwargs):
        idx = self.expert_idx
        super().add_expert(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def delete_expert(self):
        print("reset priority buffer and delete expert data")
        self._next_idx = 0
        self.expert_idx = 0
        self.count = 0

        # Pre-allocate memory
        self.actions = np.empty(self._maxsize, dtype=np.int32)
        self.diffs = np.empty(self._maxsize, dtype=np.float32)
        self.rewards = np.empty(self._maxsize, dtype=np.float32)
        self.frames = np.empty((self._maxsize, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self._maxsize, dtype=np.uint8)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.zeros((self.batch_size, self.agent_history_length, self.frame_height, self.frame_width,
                                ), dtype=np.float32)
        self.new_states = np.zeros((self.batch_size, self.agent_history_length, self.frame_height, self.frame_width,
                                    ), dtype=np.float32)
        self.indices = np.zeros(self.batch_size, dtype=np.int32)

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(self.agent_history_length, self.count - 1)
        every_range_len = p_total / batch_size
        i = 0
        while len(res) < batch_size:
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            if idx < self.agent_history_length + 1:
                continue
            if idx >= self.expert_idx and idx - self.agent_history_length < self.expert_idx:
                continue
            if idx >= self._next_idx and idx - self.agent_history_length < self._next_idx:
                continue
            if np.sum(self.terminal_flags[idx - self.agent_history_length - 1:idx - 1]) > 0:
                continue
            res.append(idx)
            i += 1
        return res

    def _sample_expert_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(self.agent_history_length, self.expert_idx - 1)
        every_range_len = p_total / batch_size
        i = 0
        while len(res) < batch_size:
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            if idx < self.agent_history_length + 1:
                continue
            if idx >= self.expert_idx and idx - self.agent_history_length < self.expert_idx:
                continue
            if np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
                continue
            res.append(idx)
            i += 1
        return res

    def sample(self, batch_size, beta, expert=False, random=False):
        """Sample a batch of experiences
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """

        assert beta > 0
        if not expert:
            idxes = self._sample_proportional(batch_size)
        else:
            idxes = self._sample_expert_proportional(batch_size)

        if random:
            weights = [1]*batch_size
            idxes = super()._get_indices(batch_size)
        else:
            weights = []
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * self.count) ** (-beta)

            for idx in idxes:
                p_sample = self._it_sum[idx] / self._it_sum.sum()
                weight = (p_sample * self.count) ** (-beta)
                weights.append(weight / max_weight)
        weights = np.array(weights)


        expert_idxes = []
        for i in range(batch_size):
            if idxes[i] < self.expert_idx:
                expert_idxes.append(1)
            else:
                expert_idxes.append(0)
        expert_idxes = np.array(expert_idxes)

        idxes = np.array(idxes)
        selected_actions = self.actions[idxes]
        selected_diffs = self.diffs[idxes]
        selected_rewards =self.rewards[idxes]
        selected_terminal = self.terminal_flags[idxes]
        for i, idx in enumerate(idxes):
            #if np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
            #    terminal_idx = idx - self.agent_history_length + np.argmax(self.terminal_flags[idx - self.agent_history_length:idx])
            #    selected_actions[i] = self.actions[terminal_idx]
            #    selected_rewards[i] = self.rewards[terminal_idx]
            #    selected_terminal[i] = self.terminal_flags[terminal_idx]
            #    self.states[i] = self._get_state(terminal_idx)
            #    self.new_states[i] = self._get_state(terminal_idx)
            #else:
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)

        # self.states = (self.states - 127.5)/127.5
        # self.new_states = (self.new_states - 127.5)/127.5
        return np.transpose(self.states, axes=(0, 2, 3, 1)), selected_actions, selected_diffs,\
               selected_rewards, np.transpose(self.new_states, axes=(0, 2, 3, 1)), selected_terminal, \
               weights, idxes, expert_idxes

    def compute_n_step_target_q(self, idxes, num_steps, gamma):
        idxes = idxes - 1
        not_terminal = np.ones((idxes.shape[0],), dtype=np.int32)
        last_step_gamma = np.zeros((idxes.shape[0],), dtype=np.float32)
        n_step_rewards = np.zeros((idxes.shape[0],), dtype=np.float32)
        n_step_state = np.zeros_like(self.states)

        for i in range(idxes.shape[0]):
            idx = idxes[i]
            n_step_idx = min(self.count-1, idx + num_steps)
            if n_step_idx >= self.expert_idx and idx < self.expert_idx:
                n_step_idx = self.expert_idx - 1
            if n_step_idx >= self._next_idx and idx  < self._next_idx:
                n_step_idx = self._next_idx - 1
            if np.sum(self.terminal_flags[idx:n_step_idx]) > 0:
                n_step_idx = idx + np.argmax(self.terminal_flags[idx:n_step_idx])
                not_terminal[i] = 0
            accum_gamma = 1
            for j in range(idx, n_step_idx):
                n_step_rewards[i] += accum_gamma * self.rewards[j]
                accum_gamma *= gamma
            last_step_gamma[i] = accum_gamma
            n_step_state[i] = self._get_state(n_step_idx)

        return n_step_rewards, np.transpose(n_step_state, axes=(0, 2, 3, 1)), last_step_gamma, not_terminal
        #return None, None, None, None


    def load_expert_data(self, path):
        data = pickle.load(open(path, 'rb'))
        num_data = len(data['frames'])
        print("Loading Expert Data ... ")
        for i in range(num_data):
            self.add_expert(obs_t=data['frames'][i], reward=data['reward'][i], action=data['actions'][i],
                            diff = self.var, done=data['terminal'][i])
            #print(data['reward'][i], np.sum(data['terminal']))
        print(self.count, "Expert Data loaded ... ")
        print(np.sum(self.diffs[:self.expert_idx]))
        print("Min Reward: ", np.min(self.rewards[self.rewards > 0]), "Max Reward: ", np.max(self.rewards[self.rewards > 0]))
        #print("Priority Buffer")
        #print(np.max(self.rewards[:self.expert_idx]))
        #quit()


    def update_priorities(self, idxes, priorities, expert_idxes, frame_num, expert_priority_decay=None, min_expert_priority=1,
                          max_prio_faction=0.005,expert_initial_priority=20):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        if expert_priority_decay is None:
            expert_priority = expert_initial_priority
        else:
            start_frame_num = 500000
            end_frame_num = 80000000
            expert_priority_decay = (expert_initial_priority-min_expert_priority)/(end_frame_num-start_frame_num)
            expert_priority = max(expert_initial_priority - max(expert_priority_decay * (frame_num-start_frame_num),0),
                                  min_expert_priority)
        #Boost expert priority as time goes on ....
        assert len(idxes) == priorities.shape[0]
        assert expert_priority >= min_expert_priority
        count = 0
        #print(expert_priority_modifier)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 0)
            assert 0 <= idx < self.count
            if expert_idxes[count] == 1:
                # update the variance of expert data
                if self.diffs[idx]==0:
                    print("zero variance")
                self.diffs[idx] = self.var/(self.var/(self.diffs[idx]+0.001)+1.0)
                priority = priority * expert_priority
                new_priority = priority #* (1 - max_prio_faction) + self._max_priority * max_prio_faction
            else:
                new_priority = priority #* (1 - max_prio_faction) + self._max_priority * max_prio_faction
            self._max_priority = max(self._max_priority, new_priority)
            #print(idx, new_priority, expert_idxes[count], count)
            self._it_sum[idx] = (new_priority) ** self._alpha
            self._it_min[idx] = (new_priority) ** self._alpha
            count += 1

