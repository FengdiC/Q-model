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
    def __init__(self, size, agent="dqn", var=1.0, bootstrap=-1, state_shape=[84, 84],
                 action_shape=None, agent_history_length=4, batch_size=32, frame_dtype=np.uint8, rescale_reward=True):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        print("Called Not Priority Queue!")
        self._maxsize = size
        self.rescale_reward = rescale_reward
        self._next_idx = 0
        self.expert_idx = 0
        self.var = var
        self.agent = agent

        self.bootstrap = bootstrap
        if self.bootstrap > 0:
            self.bootstrap_data = np.empty((self._maxsize, self.bootstrap), dtype=np.uint8)

        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.max_reward = 1
        self.min_reward = 0
        self.count = 0
        self.state_shape = state_shape

        # Pre-allocate memory
        if action_shape is None:
            self.actions = np.empty(self._maxsize, dtype=np.int32)
        else:
            self.actions = np.empty([self._maxsize] + action_shape, dtype=np.float32)
        self.diffs = np.empty(self._maxsize, dtype=np.float32)
        self.rewards = np.empty(self._maxsize, dtype=np.float32)
        self.frames = np.empty([self._maxsize] + self.state_shape, dtype=frame_dtype)
        #self.next_frames = np.empty([self._maxsize] + self.state_shape, dtype=np.uint8)
        self.terminal_flags = np.empty(self._maxsize, dtype=np.uint8)

        self.frame_indices = np.empty([self._maxsize, self.agent_history_length], dtype=np.int32)
        self.next_frame_indices = np.empty([self._maxsize, self.agent_history_length], dtype=np.int32)
        self.current_sequence_start = 0

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.zeros([self.batch_size, self.agent_history_length] + self.state_shape, dtype=np.float32)
        self.new_states = np.zeros([self.batch_size, self.agent_history_length] + self.state_shape, dtype=np.float32)

    def sample_boostrap(self, idx):
        for j in range(self.bootstrap):
            self.bootstrap_data[idx, j] = np.random.uniform(0, 1) > 0.5

    def get_idx(self):
        return self._next_idx

    def add_expert(self, obs_t, reward, action,diff, done):
        if self.rescale_reward:
            reward = np.sign(reward) * np.log(1+np.abs(reward))
        self.actions[self.expert_idx] = action
        self.diffs[self.expert_idx] = diff

        self.frames[self.expert_idx] = np.squeeze(obs_t)
        self.rewards[self.expert_idx] = reward
        self.terminal_flags[self.expert_idx] = done
        #10, 9, 8, 7
        count = 0
        for i in range(self._next_idx - self.agent_history_length + 1, self._next_idx + 1):
            i = min(max(i, self.current_sequence_start), self._maxsize)
            self.frame_indices[self._next_idx, count] = i
            count += 1
        
        old_index = self._next_idx
        self.count = min(self._maxsize, self.count + 1)
        self.expert_idx = (self.expert_idx + 1)
        if self.expert_idx > self._maxsize:
            print("Dataset too small, cannot add more expert actions .... ")
            quit()
        self._next_idx = max(self._next_idx, self.expert_idx)
        if self._next_idx != 0 and done and self.terminal_flags[old_index - 1]:
            print("expert", self._next_idx, self.count)
            self.current_sequence_start = self._next_idx
            #if sequence is completed ... 
            self.next_frame_indices[old_index] = self.frame_indices[old_index]
        else:
            count = 0
            for i in range(self._next_idx - self.agent_history_length + 1, self._next_idx + 1):
                i = min(max(i, self.current_sequence_start), self._maxsize)
                self.next_frame_indices[old_index, count] = i
                count += 1

    def __len__(self):
        return self.count

    def add(self, obs_t, reward, action, done):
        if self.rescale_reward:
            reward = np.sign(reward) * np.log(1+np.abs(reward))
        
        self.actions[self._next_idx] = action
        self.diffs[self._next_idx] = 0
        self.frames[self._next_idx] = obs_t
        self.rewards[self._next_idx] = reward
        self.terminal_flags[self._next_idx] = done
        if self.bootstrap > 0:
            self.sample_boostrap(self._next_idx)
        self.count = min(self._maxsize, self.count + 1)

        count = 0
        for i in range(self._next_idx - self.agent_history_length + 1, self._next_idx + 1):
            i = min(max(i, self.current_sequence_start), self._maxsize)
            self.frame_indices[self._next_idx, count] = i
            count += 1
        old_index = self._next_idx
        self._next_idx = (self._next_idx + 1) % (self._maxsize)
        if self._next_idx < self.expert_idx:
            self._next_idx = self.expert_idx

        if self._next_idx != 0 and done and self.terminal_flags[old_index - 1]:
             #print("expert", self._next_idx, self.count)
            self.current_sequence_start = self._next_idx
            #if sequence is completed ... 
            self.next_frame_indices[old_index] = self.frame_indices[old_index]
        else:
            count = 0
            for i in range(self._next_idx - self.agent_history_length + 1, self._next_idx + 1):
                i = min(max(i, self.current_sequence_start), self._maxsize)
                self.next_frame_indices[old_index, count] = i
                count += 1
        return old_index

    def _get_current_next_state(self, requested_index):
        # #assuming the index is for the next state
        # low_index = requested_index - self.agent_history_length
        # high_index = requested_index + 1
        # # history =4, index=10 -> 6, 7, 8, 9 ,10
        # indices_list = []
        # terminal = False
        # terminal_index = -1
        # print_flag = 0
        # count = 0
        # for index in range(low_index, high_index):
        #     #lower bounding
        #     if index < 0:
        #         index = 0
        #     #upper bounding the values .... 
        #     if high_index >= self.expert_idx and low_index < self.expert_idx:
        #         #print_flag = 1
        #         index = np.max(index, self.expert_idx)
        #     if high_index >= self._next_idx and low_index < self._next_idx:
        #         #print_flag = 2
        #         index = np.min(index, self._next_idx - 1)
            
        #     indices_list.append(index)
        #     if self.terminal_flags[index]:
        #         terminal_index = index
        #         if count < self.agent_history_length:
        #             for i in range(0, count + 1):
        #                 indices_list[i] = index + 1
        #         #print_flag = 3
        #     count += 1
        # current_states = np.zeros(self.states.shape[1:], dtype=np.float32)
        # next_states = np.zeros(self.states.shape[1:], dtype=np.float32)
        # for i in range(self.agent_history_length):
        #     current_states[i] = self.frames[indices_list[i]]
        #     next_states[i] = self.frames[indices_list[i + 1]]
        # # if print_flag:
        # #     print("flag: ", print_flag, indices_list, requested_index, terminal_index)
        # return current_states, next_states
        states = self.frames[self.frame_indices[requested_index]]
        next_states = self.frames[self.next_frame_indices[requested_index]]
        # if self.terminal_flags[requested_index]:
        #     print(self.frame_indices[requested_index], self.next_frame_indices[requested_index])
        return states, next_states

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")

        # low_index = index - self.agent_history_length + 1
        # high_index = index + 1
        # if index >= self.expert_idx and index - self.agent_history_length < self.expert_idx:
        #     continue
        # if index >= self._next_idx and index - self.agent_history_length < self._next_idx:
        #     continue
        # if np.sum(self.terminal_flags[index - self.agent_history_length:index]) > 0:
        #     continue


        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index - self.agent_history_length + 1:index + 1]

    def _get_indices(self, batch_size):
        idxes = []
        while len(idxes) < batch_size:
            index = np.random.randint(self.agent_history_length, self.count - 1)
            # if index >= self.expert_idx and index - self.agent_history_length < self.expert_idx:
            #     continue
            # if index >= self._next_idx and index - self.agent_history_length < self._next_idx:
            #     continue
            # if np.sum(self.terminal_flags[index - self.agent_history_length:index]) > 0:
            #     continue
            if index == self._next_idx - 1:
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
            index = np.random.randint(self.agent_history_length, self.expert_idx)
            # if index >= 0 and index - self.agent_history_length < self.expert_idx:
            #     continue
            # if index >= 0 and index - self.agent_history_length < self._next_idx:
            #     continue
            # if np.sum(self.terminal_flags[index - self.agent_history_length:index]) > 0:
            #      continue
            if index == self._next_idx - 1:
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

        max_reward = np.max(self.rewards)
        #Reward check
        print("Min Reward: ", np.min(self.rewards), "Max Reward: ", max_reward)
        print(self.count, "Expert Data loaded ... ")
        print("Total Rewards:", np.sum(data['reward']))
        return max_reward, self.count

    def get_bootstrap(self, idxes):
        mask = np.copy(self.bootstrap_data[idxes])
        return mask

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
            print("What is going on?", expert)
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
            states, new_states = self._get_current_next_state(idx)
            self.states[i] = states
            self.new_states[i] = new_states
            # self.states[i] = self._get_state(idx - 1)
            # self.new_states[i] = self._get_state(idx)

        if len(self.state_shape) == 2:
            states = np.transpose(self.states, axes=(0, 2, 3, 1))
            new_states = np.transpose(self.new_states, axes=(0, 2, 3, 1))
        else:
            states = self.states
            new_states = self.new_states

        return states, selected_actions,selected_diffs, \
               selected_rewards, new_states, selected_terminal


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, bootstrap=-1, state_shape=[84, 84], agent_history_length=4, agent="dqn", 
                action_shape=None, batch_size=32, frame_dtype=np.uint8, rescale_reward=True):
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
        super(PrioritizedReplayBuffer, self).__init__(size=size,agent_history_length=agent_history_length, action_shape=action_shape, 
                                                        bootstrap=bootstrap, state_shape=state_shape, frame_dtype=frame_dtype, batch_size=batch_size, rescale_reward=rescale_reward)
        assert alpha >= 0
        self._alpha = alpha
        self.agent = agent

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
        ret = super().add(*args, **kwargs)
        # if idx - self.expert_idx < self.agent_history_length:
        #     self._it_sum[idx] = 0.0001
        #     self._it_min[idx] = 0.0001
        # elif np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
        #     self._it_sum[idx] = 0.0001
        #     self._it_min[idx] = 0.0001
        # else:
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
        return ret

    def add_expert(self, *args, **kwargs):
        idx = self.expert_idx
        super().add_expert(*args, **kwargs)
        # if idx < self.agent_history_length:
        #     self._it_sum[idx] = 0.0001
        #     self._it_min[idx] = 0.0001
        # elif np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
        #     self._it_sum[idx] = 0.0001
        #     self._it_min[idx] = 0.0001
        # else:
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def delete_expert(self,size):
        self._max_priority = 5

        print("reset priority buffer and delete expert data")
        self._next_idx = 0
        self.expert_idx = 0
        self.count = 0

        # Pre-allocate memory
        self.actions = np.empty(self._maxsize, dtype=np.int32)
        self.diffs = np.empty(self._maxsize, dtype=np.float32)
        self.rewards = np.empty(self._maxsize, dtype=np.float32)
        self.frames = np.empty([self._maxsize] + self.state_shape, dtype=np.uint8)
        self.terminal_flags = np.empty(self._maxsize, dtype=np.uint8)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.zeros([self.batch_size, self.agent_history_length] + self.state_shape, dtype=np.float32)
        self.new_states = np.zeros([self.batch_size, self.agent_history_length] + self.state_shape, dtype=np.float32)
        
                
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)


    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(self.agent_history_length, self.count - 1)
        every_range_len = p_total / batch_size
        i = 0
        count = 0
        while len(res) < batch_size:
            count += 1
            if count > 10000:
                idx = np.random.randint(0, self.count)
            else:
                mass = random.random() * every_range_len + i * every_range_len
                idx = self._it_sum.find_prefixsum_idx(mass)
            if idx == self._next_idx - 1:
                continue
            # if idx < self.agent_history_length + 1:
            #     continue
            # if idx >= self.expert_idx and idx - self.agent_history_length < self.expert_idx:
            #     continue
            # if idx >= self._next_idx and idx - self.agent_history_length < self._next_idx:
            #     continue
            # if np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
            #     # if count > 9900:
            #     #     print(self.terminal_flags[idx - 25:idx + 25])
            #     continue
            res.append(idx)
            i += 1
        return res

    def _sample_expert_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(self.agent_history_length, self.expert_idx - 1)
        every_range_len = p_total / batch_size
        i = 0
        count = 0
        while len(res) < batch_size:
            count += 1
            if count > 10000:
                idx = np.random.randint(0, self.count)
            else:
                mass = random.random() * every_range_len + i * every_range_len
                idx = self._it_sum.find_prefixsum_idx(mass)
            if idx == self._next_idx - 1:
                continue
            # if idx < self.agent_history_length:
            #     continue
            # if idx >= self.expert_idx and idx - self.agent_history_length < self.expert_idx:
            #     continue
            # if np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
            #     continue
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

        total_weight = self._it_sum.sum()
        if random:
            weights = [1]*batch_size
            idxes = super()._get_indices(batch_size)
        else:
            weights = []
            p_min = self._it_min.min() / total_weight
            max_weight = (p_min * self.count) ** (-beta)

            for idx in idxes:
                p_sample = self._it_sum[idx] / total_weight
                weight = (p_sample * self.count) ** (-beta)
                weights.append(weight / max_weight)
        weights = np.array(weights)
        # if not expert and self.agent == "dqn": #Meaning pretraining ...
        #     weights = 10 * weights
        # elif not expert and self.agent == "expert":
        #     weights = 0.5 * weights

        expert_idxes = []
        for i in range(batch_size):
            if idxes[i] < self.expert_idx:
                # if self.expert_idx<4000:
                #     print("idxes[i] :::", idxes[i],":::",self.expert_idx)
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
            #self.states[i] = self._get_state(idx - 1)
            #self.new_states[i] = self._get_state(idx)
            states, new_states = self._get_current_next_state(idx)
            self.states[i] = states
            self.new_states[i] = new_states
        # self.states = (self.states - 127.5)/127.5
        # self.new_states = (self.new_states - 127.5)/127.5
        #print(idxes - 1)

        if len(self.state_shape) == 2:
            states = np.transpose(self.states, axes=(0, 2, 3, 1))
            new_states = np.transpose(self.new_states, axes=(0, 2, 3, 1))
        else:
            states = self.states
            new_states = self.new_states
        # if np.sum(selected_rewards) > 0:
        #     print(selected_rewards)
        return states, selected_actions, selected_diffs,\
               selected_rewards, new_states, selected_terminal, \
               weights, idxes, expert_idxes

    def optimized_compute_all_n_step_target_q(self, idxes, num_steps, gamma):
        not_terminal = np.ones((idxes.shape[0], num_steps), dtype=np.int32)
        last_step_gamma = np.zeros((idxes.shape[0], num_steps), dtype=np.float32)
        n_step_rewards = np.zeros((idxes.shape[0], num_steps), dtype=np.float32)
        n_step_state = np.zeros([self.states.shape[0], num_steps] + list(self.states.shape[1:]))
        n_step_actions = np.zeros([self.states.shape[0], num_steps] + list(self.actions.shape[1:]))
        last_step_gamma[:, 0] = 1

        idxes = np.array(idxes)
        last_idxes = np.array(idxes) + num_steps
        last_idxes[last_idxes > self.count - 1] = self.count - 1
        for i in range(idxes.shape[0]):
            idx = idxes[i]
            if idx >= self.expert_idx and idx < self.expert_idx:
                idx = self.expert_idx - 1
            if idx >= self._next_idx and idx  < self._next_idx:
                idx = self._next_idx - 1
            terminal_list = self.terminal_flags[idx: last_idxes[i]]           
            term_indx = np.nonzero(terminal_list)[0]
            for j in range(num_steps):
                if term_indx.shape[0] == 0:
                    idx_offset = j
                else:
                    idx_offset = min(j, term_indx[0])
                if idx + idx_offset > self.count - 1:
                    idx_offset = self.count - 1 - idx
                    
                not_terminal[i, j] = 1 - self.terminal_flags[idx + idx_offset]
                n_step_actions[i, j] = self.actions[idx + idx_offset]
                n_step_state[i, j], _ = self._get_current_next_state(idx + idx_offset)

                if idx_offset != j:
                    n_step_rewards[i, j] = n_step_rewards[i, max(0, j - 1)]
                    last_step_gamma[i, j] = last_step_gamma[i, max(0, j - 1)]

                else:
                    n_step_rewards[i, j] = gamma * n_step_rewards[i, max(0, j - 1)] + self.rewards[idx + idx_offset]
                    last_step_gamma[i, j] = last_step_gamma[i, max(0, j - 1)] * gamma
        return n_step_rewards, np.transpose(n_step_state, axes=(0, 1, 3, 4, 2)), n_step_actions,last_step_gamma, not_terminal

    def compute_all_n_step_target_q(self, idxes, num_steps, gamma):
        idxes = idxes - 1
        not_terminal = np.ones((idxes.shape[0], num_steps), dtype=np.int32)
        last_step_gamma = np.zeros((idxes.shape[0], num_steps), dtype=np.float32)
        n_step_rewards = np.zeros((idxes.shape[0], num_steps), dtype=np.float32)
        n_step_state = np.zeros([self.states.shape[0], num_steps] + list(self.states.shape[1:]))
        n_step_actions = np.zeros([self.states.shape[0], num_steps] + list(self.actions.shape[1:]))
        for i in range(idxes.shape[0]):
            idx = idxes[i]
            n_step_idx = min(self.count-1, idx + num_steps)
            if n_step_idx >= self.expert_idx and idx < self.expert_idx:
                n_step_idx = self.expert_idx - 1
            if n_step_idx >= self._next_idx and idx  < self._next_idx:
                n_step_idx = self._next_idx - 1
            if np.sum(self.terminal_flags[idx:n_step_idx]) > 0:
                for j in range(num_steps):
                    if self.terminal_flags[idx + j]:
                        n_step_idx = idx + j
                        break
            #     n_step_idx = idx + np.argmax(self.terminal_flags[idx:n_step_idx])
            #     not_terminal[i] = 0
            accum_gamma = 1
            n_range = n_step_idx - idx
            for j in range(n_range):
                accum_gamma *= gamma
                n_step_rewards[i, n_range - 1] += accum_gamma * self.rewards[idx + j]
                last_step_gamma[i, j] = accum_gamma

                n_step_rewards[i, j] = np.copy(n_step_rewards[i, n_range - 1])
                not_terminal[i, j] = 1 - self.terminal_flags[idx + j]
                n_step_state[i, j], _ = self._get_current_next_state(idx + j)
                n_step_actions[i, j] = np.copy(self.actions[idx + j])

            for j in range(n_range, num_steps):
                not_terminal[i, j] = not_terminal[i, n_range - 1]
                last_step_gamma[i, j] = last_step_gamma[i, n_range - 1]
                n_step_rewards[i, j] = n_step_rewards[i, n_range - 1]
                n_step_state[i, j] = n_step_state[i, n_range - 1]
                n_step_actions[i, j] = n_step_actions[i, n_range - 1]

        return n_step_rewards, np.transpose(n_step_state, axes=(0, 1, 3, 4, 2)), n_step_actions,last_step_gamma, not_terminal
        #return None, None, None, None


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
            n_step_state[i], _ = self._get_current_next_state(n_step_idx)
        return n_step_rewards, np.transpose(n_step_state, axes=(0, 2, 3, 1)), last_step_gamma, not_terminal
        #return None, None, None, None

    def update_priorities(self, idxes, priorities, expert_idxes, frame_num, expert_priority_modifier=1, min_priority=0.001, min_expert_priority=1,
                          expert_initial_priority=2,pretrain= False):
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

        start_frame_num = 5000000
        end_frame_num = 30000000 * expert_priority_modifier 
        base_expert_priority = (expert_initial_priority-min_expert_priority)/(end_frame_num-start_frame_num)
        expert_priority = max(expert_initial_priority - max(base_expert_priority * (frame_num-start_frame_num),0),
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
                if not pretrain:
                    self.diffs[idx] += 1
                priority = priority * expert_priority
                new_priority = priority #* (1 - max_prio_faction) + self._max_priority * max_prio_faction
            else:
                new_priority = priority #* (1 - max_prio_faction) + self._max_priority * max_prio_faction
            new_priority = new_priority * (1 - min_priority) + min_priority
            self._max_priority = max(self._max_priority, new_priority)
            #print(idx, new_priority, expert_idxes[count], count)
            self._it_sum[idx] = (new_priority) ** self._alpha
            self._it_min[idx] = (new_priority) ** self._alpha
            count += 1

