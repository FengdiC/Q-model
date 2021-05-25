
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
from env import toy_maze

def find_vars(name):
    t_vars = tf.trainable_variables()
    f_vars = [var for var in t_vars if name in var.name]
    return f_vars


def compute_regret(Q_value, grid , gamma,  final_reward=1):
    # pi = np.zeros((grid, grid), dtype=np.uint8)
    # for i in range(grid):
    #     pi[i, i] = 1

    pi = np.argmax(Q_value,axis=2)
    P = np.zeros((grid * grid, grid * grid))
    R = np.zeros((grid,grid))
    for i in range(grid-1):
        for j in range(grid-1):
            action = int(pi[i,j])
            l = i+1
            if action == 0:
                r = max(0, j - 1)
                R[i,j] = 0
            else:
                r = min(j + 1, grid - 1)
                if r == grid - 1:
                    R[i,j] = final_reward
                else:
                    R[i,j] = -0.01/grid
            P[i*grid+j,l*grid+r]=1
    for j in range(grid-1):
        P[(grid-1)*grid+j,(grid-1)*grid+j]=1
    R = np.ndarray.flatten(R)
    Q = np.matmul(np.linalg.inv(np.eye(grid*grid)-gamma*P),R)
    return Q[0]

class DQN:
    """Implements a Deep Q Network"""
    def __init__(self, args, n_actions=2, hidden=128, grid=10, name="dqn",agent='dqn'):
        """
        Args:
          n_actions: Integer, number of possible actions
          hidden: Integer, Number of filters in the final convolutional layer.
                  This is different from the DeepMind implementation
          learning_rate: Float, Learning rate for the Adam optimizer
          frame_height: Integer, Height of a frame of an Atari game
          frame_width: Integer, Width of a frame of an Atari game
          agent_history_length: Integer, Number of frames stacked together to create a state
        """
        self.args = args
        self.n_actions = n_actions
        self.hidden = hidden
        self.grid = grid
        self.var = args.var
        self.eta = args.eta
        self.agent=agent

        self.input = tf.placeholder(shape=[None, self.grid, self.grid, 1],dtype=tf.float32)
        self.weight = tf.placeholder(shape=[None,],dtype=tf.float32)
        self.policy = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.diff = tf.placeholder(shape=[None, ], dtype=tf.float32)

  
        self.q_values = self.build_network(hidden)
        # Combining value and advantage into Q-values as described above
        self.action_prob = tf.nn.softmax(self.q_values)
        self.best_action = tf.argmax(self.q_values, 1)
        # The next lines perform the parameter update. This will be explained in detail later.

        # targetQ according to Bellman equation:
        # Q = r + gamma*max Q', calculated in the function learn()
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        # Action that was performed
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        self.target_n_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.expert_state = tf.placeholder(shape=[None], dtype=tf.float32)

        # Q value of the action that was performed
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)),
                             axis=1)

        MAIN_DQN_VARS = find_vars(name)
        if self.agent == "bootdqn":
            print("Using Agent bootdqn", self.agent)
            self.loss, self.loss_per_sample = self.dqn_loss(MAIN_DQN_VARS)
        elif self.agent =="expert":
            print("Using Agent expert", self.agent)
            self.loss, self.loss_per_sample= self.expert_loss(MAIN_DQN_VARS)
            print("Using Agent ", self.agent)
        elif self.agent == "dqfd":
            print("Using Agent dqfd", self.agent)
            self.loss, self.loss_per_sample = self.dqfd_loss(MAIN_DQN_VARS)
        elif self.agent == "dqn":
            print("Using Agent dqn", self.agent)
            self.loss, self.loss_per_sample = self.dqn_loss(MAIN_DQN_VARS)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        self.update = self.optimizer.minimize(self.loss, var_list=MAIN_DQN_VARS)

    def build_network(self, hidden, index=0):
        #layers
        with tf.variable_scope('Q_network_' + str(index)):
            # layer1 = tf.layers.dense(inputs=self.input, units = hidden, activation=tf.nn.relu,
            #                             kernel_initializer=tf.variance_scaling_initializer(scale=2),name='fc1')
            # dense = tf.layers.dense(inputs=layer1, units=hidden,
            #                         kernel_initializer=tf.variance_scaling_initializer(scale=2), name="fc2")

            # f1 = tf.layers.dense(inputs=self.input, units=hidden, activation=tf.nn.relu, 
            #                         kernel_initializer=tf.glorot_normal_initializer(), name="fc1")            
            # f2 = tf.layers.dense(inputs=f1, units=hidden, activation=tf.nn.relu, 
            #                         kernel_initializer=tf.glorot_normal_initializer(), name="fc2")
            conv1 = tf.layers.conv2d(
            inputs=self.input, filters=32, kernel_size=[3, 3], strides=2,
            kernel_initializer=tf.glorot_normal_initializer(),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1')
            conv2 = tf.layers.conv2d(
            inputs=conv1, filters=64, kernel_size=[3, 3], strides=2,
            kernel_initializer=tf.glorot_normal_initializer(),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2')
            conv3 = tf.layers.conv2d(
            inputs=conv2, filters=64, kernel_size=[3, 3], strides=2,
            kernel_initializer=tf.glorot_normal_initializer(),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv3')
            # self.conv4 = tf.layers.conv2d(
            #     inputs=self.conv3, filters=hidden, kernel_size=[7, 7], strides=1,
            #     kernel_initializer=tf.variance_scaling_initializer(scale=2),
            #     padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4')
            d = tf.layers.flatten(conv3)
            dense = tf.layers.dense(inputs=d, units=hidden, activation=tf.nn.relu, 
                                    kernel_initializer=tf.glorot_normal_initializer(), name="fc3")
            # Splitting into value and advantage stream
            valuestream, advantagestream = tf.split(dense, 2, -1)
            valuestream = tf.layers.flatten(valuestream)
            advantagestream = tf.layers.flatten(advantagestream)
            advantage = tf.layers.dense(
            inputs=advantagestream, units=self.n_actions,
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name="advantage")
            value = tf.layers.dense(
            inputs=valuestream, units=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name='value')
            q_values = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))
            return q_values

    def loss_jeq(self):
        expert_act_one_hot = tf.one_hot(self.action, self.n_actions, dtype=tf.float32)
        self.expert_q_value = tf.reduce_sum(expert_act_one_hot * self.q_values, axis=1)
        #JEQ = max_{a in A}(Q(s,a) + l(a_e, a)) - Q(s, a_e)
        self.q_plus_margin = self.q_values + (1-expert_act_one_hot)*self.args.dqfd_margin
        self.max_q_plus_margin = tf.reduce_max(self.q_plus_margin, axis=1)
        jeq = (self.max_q_plus_margin - self.expert_q_value) * self.expert_state
        return jeq


    def expert_loss(self, t_vars):
        ratio = (16 + 4 * self.diff) / tf.square(4 + self.diff)

        self.prob = tf.reduce_sum(tf.multiply(self.action_prob, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)),
                               axis=1)
        self.posterior = self.target_q+ self.eta * self.var * ratio *(1-self.prob)* self.expert_state
        l_dq = tf.losses.huber_loss(labels=self.posterior, predictions=self.Q, weights=self.weight,
                                    reduction=tf.losses.Reduction.NONE)

        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq
        self.l_jeq = self.diff *(1-self.prob)/(self.target_q+0.001)

        loss_per_sample = l_dq
        loss = tf.reduce_mean(loss_per_sample+l2_reg_loss)
        return loss, loss_per_sample

    def dqfd_loss(self, t_vars):
        l_dq = tf.losses.huber_loss(labels=self.target_q, predictions=self.Q, weights=self.weight,
                                    reduction=tf.losses.Reduction.NONE)
        l_jeq = self.loss_jeq()

        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq
        self.l_jeq = l_jeq

        loss_per_sample = self.l_dq
        loss = tf.reduce_mean(loss_per_sample+self.l2_reg_loss+ self.args.LAMBDA_2 * self.l_jeq)
        return loss, loss_per_sample

    def dqn_loss(self,t_vars):
        l_dq = tf.losses.huber_loss(labels=self.target_q, predictions=self.Q, weights=self.weight, reduction=tf.losses.Reduction.NONE)
        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq
        self.l_jeq = tf.constant(0)

        loss_per_sample = l_dq
        loss = tf.reduce_mean(loss_per_sample+l2_reg_loss)
        #print("dqn_loss", t_vars)
        return loss, loss_per_sample

    def get_q_value(self, sess):
        q_values = np.zeros((self.grid, self.grid, self.n_actions), dtype=np.float32)
        for i in range(self.grid):
            for j in range(self.grid):
                #state = np.array([i,j, 0])/(0.5*self.grid)-1
                #state = np.reshape(state,(1,3))
                state = np.zeros((1, self.grid, self.grid, 1))
                state[0, i, j, 0] = 1
                value = sess.run(self.q_values, feed_dict={self.input:state})
                q_values[i, j] = value[0]
        return q_values

def learn(session, states, actions, diffs, rewards, new_states, terminal_flags,
            weights, expert_idxes, main_dqn, target_dqn, batch_size, gamma, agent,shaping=None):
    """
    Args:
        session: A tensorflow sesson object
        replay_memory: A ReplayMemory object
        main_dqn: A DQN object
        target_dqn: A DQN object
        batch_size: Integer, Batch size
        gamma: Float, discount factor for the Bellman equation
    Returns:
        loss: The loss of the minibatch, for tensorboard
    Draws a minibatch from the replay memory, calculates the
    target Q-value that the prediction Q-value is regressed to.
    Then a parameter update is performed on the main DQN.
    """
    states = np.expand_dims(np.squeeze(states), axis=3)
    new_states = np.expand_dims(np.squeeze(new_states), axis=3)
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})
    double_q = q_vals[range(batch_size), arg_q_max]

    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = rewards + (gamma*double_q *  (1-terminal_flags))
    if agent == 'shaping':
        state_indices = np.round(((states+1)/2.0 * main_dqn.grid)).astype(np.uint8)
        new_state_indices = np.round(((new_states+1)/2.0 * main_dqn.grid)).astype(np.uint8)
        current_potential = shaping[state_indices[:,0].astype(np.uint8),state_indices[:,1].astype(np.uint8)]
        next_potential = shaping[new_state_indices[:,0].astype(np.uint8),new_state_indices[:,1].astype(np.uint8)]
        curr= current_potential[range(batch_size),actions]
        next = next_potential[range(batch_size),arg_q_max]
        target_q += gamma* next*(1-terminal_flags) - curr

    loss_sample, l_dq, l_jeq, _ = session.run([main_dqn.loss_per_sample, main_dqn.l_dq,
                                                main_dqn.l_jeq, main_dqn.update],
                          feed_dict={main_dqn.input:states,
                                     main_dqn.target_q:target_q,
                                     main_dqn.action:actions,
                                     main_dqn.expert_state:expert_idxes,
                                     main_dqn.weight:weights,
                                     main_dqn.diff: diffs
                                     })
    # for i in range(batch_size):
    #     if loss_sample > 0.05:
    #         print(i, loss_sample[i], target_q[i], r[i], np.mean(loss_sample))
    return loss_sample, np.mean(l_dq), np.mean(l_jeq)

class toy_env:
    def __init__(self, grid, final_reward=1, args=None, min_expert_frames=512, expert=True):
        self.grid = grid
        self.final_reward = final_reward
        self.final = False
        self.n_actions=2
        self.reset()
        if expert:
            self.generate_expert_data(min_expert_frames=min_expert_frames, args=args)

    def reset(self):
        self.current_state_x = 0
        self.current_state_y = 0
        self.timestep = 0
        state = np.zeros((self.grid, self.grid))
        state[self.current_state_x, self.current_state_y] = 1
        return state

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

        if (self.current_state_x >= self.grid - 1) and (self.current_state_y >= self.grid - 1):
            reward = self.final_reward
            self.final=True
        self.timestep += 1
        if self.timestep >= self.grid - 1:
            terminal = 1
        else:
            terminal = 0
        state = np.zeros((self.grid, self.grid))
        state[int(self.current_state_x), int(self.current_state_y)] = 1
        return state, reward, terminal
        #return np.array([self.current_state_x/self.grid, self.current_state_y/self.grid, self.timestep/self.grid]) * 2 - 1, reward, terminal


    def generate_expert_data(self, min_expert_frames=512, expert_ratio=1, args=None):
        expert = {}
        half_expert_traj = (self.grid)//expert_ratio
        num_batches = math.ceil(min_expert_frames/half_expert_traj)
        num_expert = num_batches * half_expert_traj
        
        expert_frames = []
        rewards = []
        terminals = []
        current_index = 0
        for i in range(num_batches):
            current_state = self.reset()
            for j in range(half_expert_traj):
                expert_frames.append(current_state)
                s, r, t = self.step(1)
                current_state = s
                rewards.append(r)
                terminals.append(t)
                if t:
                    expert_frames.append(current_state)
                    rewards.append(0)
                    terminals.append(t)
                    break
        expert_frames = np.array(expert_frames).astype(np.float32)
        rewards = np.array(rewards).astype(np.float32)
        terminals = np.array(terminals).astype(np.float32)

        expert['actions'] = np.ones((num_expert,), dtype=np.uint8)
        expert['frames'] = expert_frames
        expert['reward'] = rewards
        expert['terminal'] = terminals
        self.final = False
        print("Creating Expert Data ... ", current_index)
        with open(args.expert_dir+args.expert_file, 'wb') as fout:
            pickle.dump(expert, fout)

    def print_state(self):
        print(self.current_state_x+1, self.current_state_y+1)

    def get_default_shape(self):
        return [self.grid, self.grid]

def eval_env(sess, args, env, ensemble, frame_num, eps_length, action_getter,grid, pretrain=False, index=-1):
    if index == -1:
        MAIN_DQN = ensemble[np.random.randint(0, len(bootstrap_dqns))]["main"]
    else:
        MAIN_DQN = ensemble[index]["main"]    
    episode_reward_sum = 0
    episode_len = 0
    eval_pos = []
    next_frame = env.reset()
    eval_pos.append(np.argmax(next_frame))
    for _ in range(eps_length):
        action = action_getter.get_action(sess, frame_num, next_frame, MAIN_DQN, evaluation=True)
        next_frame, reward, terminal = env.step(action)
        episode_reward_sum += reward
        episode_len += 1
        eval_pos.append(np.argmax(next_frame))
        if terminal:
            break
    return episode_reward_sum, episode_len, eval_pos

def build_initial_replay_buffer(sess, env, replay_buffer, action_getter, max_eps_length, replay_buf_size, args):
    frame_num = 0
    while frame_num < replay_buf_size:
        frame = env.reset()
        for _ in range(max_eps_length):
            # 
            action = action_getter.get_random_action()
            # print("Action: ",action)
            # 
            next_frame, reward, terminal = env.step(action)
            #  Store transition in the replay memory
            replay_buffer.add(obs_t=frame, reward=reward, action=action, done=terminal)
            if terminal:
                replay_buffer.add(obs_t=next_frame, reward=0, action=action, done=terminal)
            frame = next_frame
            frame_num += 1
            if terminal:
                break

def train_step_bootdqn(sess, args, env, bootstrap_dqns, replay_buffer, frame_num, eps_length,
                    learn, action_getter,grid,agent, pretrain=False):
    start_time = time.time()
    episode_reward_sum = 0
    episode_length = 0
    episode_loss = []

    episode_dq_loss = []
    episode_jeq_loss = []

    expert_ratio = []

    NETW_UPDATE_FREQ = 25 * grid        # Number of chosen actions between updating the target network.
    DISCOUNT_FACTOR = args.gamma           # gamma in the Bellman equation
    UPDATE_FREQ = args.update_freq                  # Every four actions a gradient descend step is performed
    BS = 32
    terminal = False
    frame = env.reset()
    #select a dqn ... 
    selected_dqn = bootstrap_dqns[np.random.randint(0, len(bootstrap_dqns))]
    priority = np.zeros((BS, ), dtype=np.float32)
    priority_weight = np.zeros((BS,), dtype=np.float32)
    for _ in range(eps_length):
        if not pretrain:
            if args.stochastic_exploration == "True":
                action = action_getter.get_stochastic_action(sess, frame, selected_dqn["main"])
            else:
                action = action_getter.get_action(sess, frame_num, frame, selected_dqn["main"],evaluation=True)
            next_frame, reward, terminal = env.step(action)
            replay_buffer.add(obs_t=frame, reward=reward, action=action, done=terminal)
            if terminal:
                replay_buffer.add(obs_t=next_frame, reward=0, action=action, done=terminal)
            frame = next_frame
            episode_length += 1
            episode_reward_sum += reward
        frame_num += 1

        if frame_num % UPDATE_FREQ == 0 and frame_num > grid-1:
            generated_states, generated_actions, generated_diffs, generated_rewards, generated_new_states, \
            generated_terminal_flags, generated_weights, idxes, expert_idxes = replay_buffer.sample(
                BS, args.beta, expert=pretrain)  # Generated trajectories
            bootstrap_mask = replay_buffer.get_bootstrap(idxes)

            for bootstrap_dqn in bootstrap_dqns:
                MAIN_DQN = bootstrap_dqn["main"]
                TARGET_DQN = bootstrap_dqn["target"]
                mask = bootstrap_mask[:, bootstrap_dqn["idx"]]
                generated_weights = generated_weights * mask

                loss, loss_dq, loss_jeq = learn(sess, generated_states, generated_actions, generated_diffs,generated_rewards,
                                                       generated_new_states, generated_terminal_flags, generated_weights,
                                                       expert_idxes, MAIN_DQN, TARGET_DQN, BS,DISCOUNT_FACTOR, agent)
                episode_dq_loss.append(loss_dq)
                episode_jeq_loss.append(loss_jeq)

                priority += loss
                priority_weight += mask
            priority_weight = np.clip(priority_weight, 1, BS * 2)
            priority = priority/priority_weight
            replay_buffer.update_priorities(idxes, priority, expert_idxes, frame_num, expert_priority_modifier=args.expert_priority_modifier,
                                            min_expert_priority=args.min_expert_priority,pretrain = pretrain)
            expert_ratio.append(np.sum(expert_idxes)/BS)
            episode_loss.append(loss)
        if pretrain:
            if frame_num % (NETW_UPDATE_FREQ//UPDATE_FREQ) == 0 and frame_num > 0:
                print("UPDATING Network ... ")
                for bootstrap_dqn in bootstrap_dqns:
                    network_updater = bootstrap_dqn["updater"]
                    network_updater.update_networks(sess)
        else:
            if frame_num % NETW_UPDATE_FREQ == 0 and frame_num > 0:
                print("UPDATING Network ... ", frame_num)
                for bootstrap_dqn in bootstrap_dqns:
                    network_updater = bootstrap_dqn["updater"]
                    network_updater.update_networks(sess)
            if terminal:
                break
    return episode_reward_sum, episode_length, np.mean(episode_loss), np.mean(episode_dq_loss), \
           np.mean(episode_jeq_loss), time.time() - start_time, np.mean(expert_ratio)


def train_bootdqn(priority=True, agent='model', num_bootstrap=20,seed=0,grid=10):
    tf.reset_default_graph()
    args = utils.argsparser()
    name = agent
    args.seed=seed
    args.lr = 0.05
    tf.random.set_random_seed(args.seed)
    np.random.seed(args.seed)
    NETW_UPDATE_FREQ = 25 * grid        # Number of chosen actions between updating the target network.

    MAX_EPISODE_LENGTH = grid*grid

    REPLAY_MEMORY_START_SIZE = 32 * 110 # Number of completely random actions,
    # before the agent starts learning
    MAX_FRAMES = 50000000  # Total number of frames the agent sees
    MEMORY_SIZE = 32 * 2200#grid * grid +2 # Number of transitions stored in the replay memory
    # evaluation episode
    HIDDEN = 512
    BS = 32
    # main DQN and target DQN networks:
    print("Agent: ", name)

    if args.env_id == 'maze':
        env = toy_maze('mazes')
    else:
        final_reward = 1
        env = toy_env(grid, final_reward, args=args)

    bootstrap_dqns = []
    for i in range(num_bootstrap):
        print("Making network ", i)
        with tf.variable_scope('mainDQN_' + str(i)):
            MAIN_DQN = DQN(args, env.n_actions, HIDDEN,grid=grid, name="mainDQN_" + str(i),agent=agent)
        with tf.variable_scope('targetDQN_' + str(i)):
            TARGET_DQN = DQN(args, env.n_actions, HIDDEN, grid=grid, name="targetDQN_" + str(i),agent=agent)
        MAIN_DQN_VARS = find_vars("mainDQN_" + str(i))
        TARGET_DQN_VARS = find_vars("targetDQN_" + str(i))
        network_updater = utils.TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
        bootstrap_dqns.append({"main":MAIN_DQN, "target":TARGET_DQN, "updater":network_updater, "idx":i})

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if priority:
        print("Priority", grid, grid * grid)
        my_replay_memory = PriorityBuffer.PrioritizedReplayBuffer(MEMORY_SIZE, args.alpha, state_shape=[args.state_size], 
                                                                agent_history_length=1, agent=name, batch_size=BS, 
                                                                frame_dtype=np.float32, bootstrap=num_bootstrap)
    else:
        print("Not Priority")
        my_replay_memory = PriorityBuffer.ReplayBuffer(MEMORY_SIZE, state_shape=[args.state_size],agent_history_length=1, 
                                                        frame_dtype=np.float32, agent=name, batch_size=BS, bootstrap=num_bootstrap)
    action_getter = utils.ActionGetter(env.n_actions,eps_annealing_frames=MEMORY_SIZE,
                                    replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                    max_frames=MAX_FRAMES,
                                    eps_initial=args.initial_exploration)
    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session(config=config)
    sess.run(init)
    eps_number = 0
    frame_number = 0

    print("Agent: ", name)
    last_eval = 0
    build_initial_replay_buffer(sess, env, my_replay_memory, action_getter, MAX_EPISODE_LENGTH, REPLAY_MEMORY_START_SIZE, args)
    max_eps = 500
    regret_list = []

    #compute regret
    # Q_value = np.zeros((grid, grid, 2))
    # Q_value[:, :, 1] = final_reward
    # V = compute_regret(Q_value, grid, args.gamma,final_reward)
    while frame_number < MAX_FRAMES:
        eps_rw, eps_len, eps_loss, eps_dq_loss, eps_jeq_loss, eps_time, exp_ratio = train_step_bootdqn(
            sess, args, env, bootstrap_dqns, my_replay_memory,  frame_number,
            MAX_EPISODE_LENGTH, learn, action_getter, grid, agent,pretrain=False)
        frame_number += eps_len
        eps_number += 1
        last_eval += eps_len
        # print("GridSize", grid, "EPS: ", eps_number, "Mean Reward: ", eps_rw, "seed", args.seed)
        if args.env_id=='chain':
            selected_dqn = bootstrap_dqns[np.random.randint(0, len(bootstrap_dqns))]
            MAIN_DQN = selected_dqn["main"]
            TARGET_DQN = selected_dqn["target"]
            q_values = MAIN_DQN.get_q_value(sess)
            pi = np.argmax(q_values, axis=2)
            correct = grid - 1 - np.sum(np.diag(pi)[:-1])
            # correct = np.sum(pi[:, 0])
            print(grid , eps_number, correct, eps_rw)
            regret_list.append(correct)
            # #compute regret
            # regret_list.append(V - compute_regret(q_values, grid, args.gamma, final_reward))
            # print(V, eps_number, regret_list[-1], eps_rw)
            if (len(regret_list)>5 and np.mean(regret_list[-3:]) < 0.02 and env.final) or eps_number > max_eps:
                print("GridSize", grid, "EPS: ", eps_number, "Mean Reward: ", eps_rw, "seed", args.seed)
                return eps_number

def potential(grid):
    potential = np.zeros((grid,grid,2))
    for j in reversed(range(grid+1)):
        for i in range(j):
            potential[i,i+grid-j,1] = np.exp(-0.5*(grid-j)**2)
            potential[i + grid - j,i, 1] = np.exp(-0.5 * (grid - j) ** 2)
    # potential = np.reshape(potential,(grid*grid,2))
    return potential

def train_step_dqfd(sess, args, env, MAIN_DQN, TARGET_DQN, network_updater, replay_buffer, frame_num, eps_length,
                    learn, action_getter, grid, shaping,agent,pretrain=False, temporal=True, NETW_UPDATE_FREQ=512):
    start_time = time.time()
    episode_reward_sum = 0
    episode_length = 0
    episode_loss = []
    regret_list = []
    frame_list = []

    episode_dq_loss = []
    episode_jeq_loss = []

    expert_ratio = []

    DISCOUNT_FACTOR = args.gamma  # gamma in the Bellman equation
    UPDATE_FREQ = args.update_freq  # Every four actions a gradient descend step is performed
    BS = 32
    terminal = False
    frame = env.reset()
    for _ in range(eps_length):
        if not pretrain:
            if args.stochastic_exploration == "True":
                action = action_getter.get_stochastic_action(sess, frame, MAIN_DQN)
            # elif agent!='dqn':
            #     action = action_getter.get_action(sess, frame_num, frame, MAIN_DQN, evaluation=True)
            else:
                action = action_getter.get_action(sess, frame_num, frame, MAIN_DQN, evaluation=False, temporal=temporal)
            next_frame, reward, terminal = env.step(action)
            replay_buffer.add(obs_t=frame, reward=reward, action=action, done=terminal)
            if terminal:
                replay_buffer.add(obs_t=next_frame, reward=0, action=action, done=terminal)
            frame = next_frame

            episode_length += 1
            episode_reward_sum += reward
        frame_num += 1

        if frame_num % UPDATE_FREQ == 0 and frame_num > grid - 1:
            generated_states, generated_actions, generated_diffs, generated_rewards, generated_new_states, \
            generated_terminal_flags, generated_weights, idxes, expert_idxes = replay_buffer.sample(
                BS, args.beta, expert=pretrain)  # Generated trajectories
            loss, loss_dq, loss_jeq = learn(sess, generated_states, generated_actions, generated_diffs,
                                            generated_rewards,
                                            generated_new_states, generated_terminal_flags, generated_weights,
                                            expert_idxes, MAIN_DQN, TARGET_DQN, BS, DISCOUNT_FACTOR, agent,shaping)
            episode_dq_loss.append(loss_dq)
            episode_jeq_loss.append(loss_jeq)

            replay_buffer.update_priorities(idxes, loss, expert_idxes, frame_num,
                                            expert_priority_modifier=args.expert_priority_modifier,
                                            min_expert_priority=args.min_expert_priority, pretrain=pretrain)
            expert_ratio.append(np.sum(expert_idxes) / BS)
            episode_loss.append(loss)
        if pretrain:
            if frame_num % (NETW_UPDATE_FREQ // UPDATE_FREQ) == 0 and frame_num > 0:
                print("UPDATING Network ... ")
                network_updater.update_networks(sess)
        else:
            if frame_num % NETW_UPDATE_FREQ == 0 and frame_num > 0:
                print("UPDATING Network ... ", frame_num)
                network_updater.update_networks(sess)
            if terminal:
                break
    return episode_reward_sum, episode_length, np.mean(episode_loss), np.mean(episode_dq_loss), \
           np.mean(episode_jeq_loss), time.time() - start_time, np.mean(expert_ratio), regret_list, frame_list

def potential_pretrain(session, states, actions, diffs, rewards, new_states, terminal_flags,
            weights, expert_idxes, main_dqn, target_dqn, batch_size, gamma, agent,shaping=None):

    states = np.squeeze(states)
    state_indices = np.round(((states+1)/2.0 * main_dqn.grid)).astype(np.uint8)
    current_potential = shaping[state_indices[:, 0].astype(np.uint8), state_indices[:, 1].astype(np.uint8)]
    target_q = current_potential[range(batch_size), actions]
    loss_sample, l_dq, l_jeq, _ = session.run([main_dqn.loss_per_sample, main_dqn.l_dq,
                                                main_dqn.l_jeq, main_dqn.update],
                          feed_dict={main_dqn.input:states,
                                     main_dqn.target_q:target_q,
                                     main_dqn.action:actions,
                                     main_dqn.expert_state:expert_idxes,
                                     main_dqn.weight:weights,
                                     main_dqn.diff: diffs
                                     })
    return loss_sample, np.mean(l_dq), np.mean(l_jeq)

def train(tflogger, priority=True, agent='model', grid=10, seed=0, max_len=600, env_mode=0, temporal=True, 
            NETW_UPDATE_FREQ=512):
    tf.reset_default_graph()
    with tf.variable_scope(agent):
        args = utils.argsparser()
        args.seed = seed
        name = agent
        # if agent=='expert':
        #     args.lr=0.0005
        # else:
        #     args.lr=0.05
        tf.random.set_random_seed(args.seed)
        np.random.seed(args.seed)
        shaping = potential(grid)

        MAX_EPISODE_LENGTH = grid*grid

        REPLAY_MEMORY_START_SIZE = 500  # Number of completely random actions,
        # before the agent starts learning
        MAX_FRAMES = 25000  # Total number of frames the agent sees
        MEMORY_SIZE = 10000  # grid * grid +2 # Number of transitions stored in the replay memory
        # evaluation episode
        HIDDEN = 512
        BS = 32
        # main DQN and target DQN networks:
        print("Agent: ", name)
        
        if env_mode == 0:
            final_reward = 1 #1, -1
        else:
            final_reward = -1
        env = toy_env(grid, final_reward, args=args)

        with tf.variable_scope('mainDQN'):
            MAIN_DQN = DQN(args, env.n_actions, HIDDEN, grid=grid, name="mainDQN",agent=agent)
        with tf.variable_scope('targetDQN'):
            TARGET_DQN = DQN(args, env.n_actions, HIDDEN,  grid=grid, name="targetDQN",agent=agent)

        init = tf.global_variables_initializer()
        MAIN_DQN_VARS = find_vars('mainDQN')
        TARGET_DQN_VARS = find_vars('targetDQN')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        if priority:
            print("Priority", grid, grid * grid)
            my_replay_memory = PriorityBuffer.PrioritizedReplayBuffer(MEMORY_SIZE*20, args.alpha,
                                                                      state_shape=[grid_size, grid_size], agent_history_length=1,
                                                                      agent=name, frame_dtype=np.float32, batch_size=BS, rescale_reward=False)
        else:
            print("Not Priority")
            my_replay_memory = PriorityBuffer.ReplayBuffer(MEMORY_SIZE*20, state_shape=[grid_size, grid_size],
                                                           agent_history_length=1, agent=name, frame_dtype=np.float32, 
                                                           batch_size=BS, rescale_reward=False)
        network_updater = utils.TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
        action_getter = utils.ActionGetter(env.n_actions,eps_annealing_frames=MEMORY_SIZE,
                                           replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                           max_frames=MAX_FRAMES,
                                           eps_initial=args.initial_exploration)
        saver = tf.train.Saver(max_to_keep=10)
        sess = tf.Session(config=config)
        sess.run(init)
        eps_number = 0
        frame_number = REPLAY_MEMORY_START_SIZE

        my_replay_memory.load_expert_data(args.expert_dir+args.expert_file)

        print("Agent: ", name)
        regret_list = []
        max_eps = max_len
        # # compute regret
        # Q_value = np.zeros((grid, grid, 2))
        # Q_value[:, :, 1] = final_reward
        # V = compute_regret(Q_value, grid, args.gamma, final_reward)
        # print("True value for initial state: ", V)
        last_eval = 0
        # if agent !='dqn':
        if agent == 'shaping':
            print("Beginning to pretrain shaping", my_replay_memory.count)
            train_step_dqfd(
                sess, args, env, MAIN_DQN, TARGET_DQN, network_updater, my_replay_memory, frame_number,
                args.pretrain_bc_iter, potential_pretrain, action_getter, grid, shaping, agent, 
                pretrain=True, temporal=temporal, NETW_UPDATE_FREQ=NETW_UPDATE_FREQ)
            print("done pretraining ,test prioritized buffer, shaping")
            print("buffer expert size: ", my_replay_memory.expert_idx)
        else:
            # print("Beginning to pretrain")
            # train_step_dqfd(
            #     sess, args, env, MAIN_DQN, TARGET_DQN, network_updater, my_replay_memory, frame_number,
            #     args.pretrain_bc_iter, learn, action_getter, grid, shaping, agent,
            #     pretrain=True, temporal=temporal, NETW_UPDATE_FREQ=NETW_UPDATE_FREQ)
            # print("done pretraining ,test prioritized buffer")
            # print("buffer expert size: ", my_replay_memory.expert_idx)
            print("No Pretraining ... ")
        # else:
        #     print("Expert data deleted .... ")
        #     my_replay_memory.delete_expert(MEMORY_SIZE)

        build_initial_replay_buffer(sess, env, my_replay_memory, action_getter, MAX_EPISODE_LENGTH,
                                        REPLAY_MEMORY_START_SIZE,args)



        eps_return_list = []
        eps_length_list = []
        eps_loss_list = []
        eps_eval_return_list = []
        if temporal:
            tf_name = "model"
        else:
            tf_name = "model"
        while eps_number < max_eps:
            if eps_number % 20 == 0:
                episode_reward_sum = 0
                next_frame = env.reset()
                for _ in range(MAX_EPISODE_LENGTH):
                    action = action_getter.get_action(sess, 0, next_frame, MAIN_DQN, evaluation=True)
                    next_frame, reward, terminal = env.step(action)
                    episode_reward_sum += reward
                    if terminal:
                        break
                eps_eval_return_list.append(episode_reward_sum)
                tflogger.log_scalar("Evaluation/Reward_" + tf_name, episode_reward_sum, eps_number)
                #q_values = MAIN_DQN.get_q_value(sess)
                # plt.imshow(q_values[:, :, 0], cmap='hot', interpolation='nearest')
                # plt.savefig("figures/a_0_" +  str(agent) + ".png")
                # plt.close()
                # plt.imshow(q_values[:, :, 1], cmap='hot', interpolation='nearest')
                # plt.savefig("figures/a_1_"  +  str(agent) + ".png")
                # plt.close()

                # q_values = TARGET_DQN.get_q_value(sess)
                # plt.imshow(q_values[:, :, 0], cmap='hot', interpolation='nearest')
                # plt.savefig("figures/t_a_0_" +  str(agent) + ".png")
                # plt.close()
                # plt.imshow(q_values[:, :, 1], cmap='hot', interpolation='nearest')
                # plt.savefig("figures/t_a_1_" +  str(agent) + ".png")
                # plt.close()

            eps_rw, eps_len, eps_loss, eps_dq_loss, eps_jeq_loss, eps_time, exp_ratio, _, _ = train_step_dqfd(
                sess, args, env, MAIN_DQN, TARGET_DQN, network_updater, my_replay_memory, frame_number,
                MAX_EPISODE_LENGTH, learn, action_getter, grid,shaping, agent,pretrain=False, NETW_UPDATE_FREQ=NETW_UPDATE_FREQ)
            frame_number += eps_len
            eps_number += 1
            last_eval += eps_len
            # print("GridSize", grid, "EPS: ", eps_number, "Mean Reward: ", eps_rw, "seed", args.seed,'EPS Length: ',eps_len,
            #       "Level: ",env.level)

            if args.env_id == 'chain':
                q_values = MAIN_DQN.get_q_value(sess)
                pi = np.argmax(q_values, axis=2)
                # print(np.diag(pi)[:-1])
                if final_reward == 1:
                    correct = grid -1 - np.sum(np.diag(pi)[:-1])
                else:
                    correct = np.sum(pi[:,0])
                print(grid , eps_number, correct,eps_rw)
                regret_list.append(correct)
                eps_return_list.append(eps_rw)
                eps_length_list.append(eps_len)
                eps_loss_list.append(eps_loss)
                
                tflogger.log_scalar("Training/Reward_" + tf_name, eps_rw, eps_number)
                tflogger.log_scalar("Training/Loss_" + tf_name, eps_loss, eps_number)
                tflogger.log_scalar("Training/Regret_" + tf_name, correct, eps_number)
                # compute regret
                # regret_list.append(V - compute_regret(q_values, grid, args.gamma,  final_reward))
                # print(V, eps_number, regret_list[-1], eps_rw)
                # regret_list.append(eps_rw)
                #if (len(regret_list)>5 and np.mean(regret_list[-3:]) < 0.02 and env.final) or eps_number > max_eps:
                #print("GridSize", grid, "EPS: ", eps_number, "Mean Reward: ", eps_rw, "seed", args.seed) #possibly modify the criterion
                #return eps_number #reward, regret, 


        return np.array(eps_return_list), np.array(eps_length_list), np.array(regret_list), np.array(eps_loss_list), np.array(eps_eval_return_list)
# train_bootdqn(grid=20,agent='bootdqn')
# train(grid=55,agent='dqfd')
import matplotlib.pyplot as plt
idx = range(50,80,3)
max_len = 600
grid_size = 48
NETW_UPDATE_FREQ=256
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
args = utils.argsparser()
tflogger = tensorflowboard_logger("logs/toy_chain_seed_" +  str(args.seed) + "/" + args.agent + "_mode_" +  str(args.env_mode) +  "/", sess, None)
train(tflogger, grid=grid_size,agent=args.agent,seed=args.seed, max_len=max_len, env_mode=args.env_mode, temporal=args.use_ezgreedy, NETW_UPDATE_FREQ=NETW_UPDATE_FREQ)
#train(tflogger, grid=grid_size,agent="dqn",seed=1, max_len=max_len, env_mode=final_mode, temporal=temporal, NETW_UPDATE_FREQ=NETW_UPDATE_FREQ)
#train(tflogger, grid=grid_size,agent="dqn",seed=2, max_len=max_len, env_mode=final_mode, temporal=temporal, NETW_UPDATE_FREQ=NETW_UPDATE_FREQ)
# dqn_eps_ret = (dqn_eps_ret1 + dqn_eps_ret2 + dqn_eps_ret3)
# dqn_eps_len = (dqn_eps_len1 + dqn_eps_len2 + dqn_eps_len3)
# dqn_eps_regret = (dqn_regret1 + dqn_regret2 + dqn_regret3)
# dqn_eps_loss = (dqn_loss1 + dqn_loss2 + dqn_loss3)
# plt.plot(np.arange(max_len), dqn_eps_ret, label="DQN Returns")
# plt.xlabel("Training step")
# plt.ylabel("Reward")
# plt.title("Training Reward")
# plt.legend()
# plt.savefig("figures/dqn_result_rew.png")
# plt.close()
# #dqn with expert, bqfd, dqfd
# train(tflogger, grid=grid_size,agent="dqfd",seed=0, max_len=max_len, env_mode=final_mode, temporal=temporal, NETW_UPDATE_FREQ=NETW_UPDATE_FREQ)
# train(tflogger, grid=grid_size,agent="dqfd",seed=1, max_len=max_len, env_mode=final_mode, temporal=temporal, NETW_UPDATE_FREQ=NETW_UPDATE_FREQ)
# train(tflogger,grid=grid_size,agent="dqfd",seed=2, max_len=max_len, env_mode=final_mode, temporal=temporal, NETW_UPDATE_FREQ=NETW_UPDATE_FREQ)
# dqfd_eps_ret = (dqfd_eps_ret1 + dqfd_eps_ret2 + dqfd_eps_ret3)
# dqfd_eps_len = (dqfd_eps_len1 + dqfd_eps_len2 + dqfd_eps_len3)
# dqfd_eps_regret = (dqfd_regret1 + dqfd_regret2 + dqfd_regret3)
# dqfd_eps_loss = (dqfd_loss1 + dqfd_loss2 + dqfd_loss3)
# print(dqfd_eps_ret)

# train(tflogger, grid=grid_size,agent="expert",seed=0, max_len=max_len, env_mode=final_mode, temporal=temporal, NETW_UPDATE_FREQ=NETW_UPDATE_FREQ)
# train(tflogger, grid=grid_size,agent="expert",seed=1, max_len=max_len, env_mode=final_mode, temporal=temporal, NETW_UPDATE_FREQ=NETW_UPDATE_FREQ)
# train(tflogger, grid=grid_size,agent="expert",seed=2, max_len=max_len, env_mode=final_mode, temporal=temporal, NETW_UPDATE_FREQ=NETW_UPDATE_FREQ)

# plt.plot(np.arange(max_len), dqfd_eps_ret, label="DQFD Returns")
# plt.xlabel("Training step")
# plt.ylabel("Reward")
# plt.title("Training Reward")
# plt.legend()
# plt.savefig("figures/dqfd_result_rew.png")
# plt.close()

# plt.plot(np.arange(max_len), dqn_eps_ret, label="DQN Returns")
# plt.plot(np.arange(max_len), dqfd_eps_ret, label="DQFD Returns")
# plt.plot(np.arange(max_len), bqfd_eps_ret, label="BQFD Returns")
# plt.xlabel("Training step")
# plt.ylabel("Reward")
# plt.title("Training Reward")
# plt.legend()
# plt.savefig("figures/result_rew.png")
# plt.close()

# plt.plot(np.arange(max_len), dqn_eps_regret, label="DQN Regret")
# plt.plot(np.arange(max_len), dqfd_eps_regret, label="DQFD Regret")
# plt.plot(np.arange(max_len), bqfd_eps_regret, label="BQFD Regret")
# plt.xlabel("Training step")
# plt.ylabel("Len")
# plt.title("Regret")
# plt.legend()
# plt.savefig("figures/result_regret.png")
# plt.close()

# plt.plot(np.arange(max_len), dqn_eps_loss, label="DQN Loss")
# plt.plot(np.arange(max_len), dqfd_eps_loss, label="DQFD Loss")
# plt.plot(np.arange(max_len), bqfd_eps_loss, label="BQFD Loss")
# plt.xlabel("Training step")
# plt.ylabel("Len")
# plt.title("Loss")
# plt.legend()
# plt.savefig("figures/result_loss.png")
# plt.close()

# Compare bewteen all algorithms
# num_runs = 10
# reach = np.zeros((5, num_runs))
# for seed in range(3):
#     for grid_index in range(num_runs):
#         grid = 3 * grid_index + 50
#         print("epsilon: grid_",grid,"seed_",seed)
#         #current_time = time.time()
#         num_potential = train(grid=grid,agent='shaping',seed=seed)
#         num_boot = train_bootdqn(grid=grid,agent='bootdqn',seed=seed)
#         #print("Elapsed: ", time.time() - current_time)
#         num_dqn = train(grid=grid,agent='dqn',seed=seed)
#         reach[0,grid_index] += num_dqn
#         reach[1,grid_index] += num_boot
#
#         num = train(grid=grid,agent='expert',seed=seed)
#         num_dqfd= train(grid=grid,agent='dqfd',seed=seed)
#         reach[3,grid_index] += num_dqfd
#         reach[4,grid_index] += num_potential
#         reach[2,grid_index] += num
#     np.save('bootdqn_explor',reach)
#     # plt.plot(range(50,80,3),reach[0,:num_runs],label='DQN with EZ greedy',color='b')
#     # plt.plot(range(50,80,3),reach[1,:num_runs],label='bootstrapped DQN',color='m')
#     # plt.plot(range(50,80,3),reach[3,:num_runs],label='DQfD',color='tab:orange')
#     # plt.plot(range(50,80,3),reach[4,:num_runs],label='RLfD through shaping',color='y')
#     # plt.plot(range(50,80,3),reach[2,:num_runs],label='BQfD',color='r')
#     # plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1))
#     # plt.tight_layout()
#     # plt.savefig('chain_rlfd')
#
# reach = reach/3.0
# np.save('bootdqn_explor',reach)
#
# # reach =np.load('bootdqn_explor.npy')
# plt.plot(range(50,80,3),reach[0,:num_runs],label='DQN with EZ greedy',color='b')
# plt.plot(range(50,80,3),reach[1,:num_runs],label='bootstrapped DQN',color='m')
# plt.plot(range(50,80,3),reach[3,:num_runs],label='DQfD',color='tab:orange')
# plt.plot(range(50,80,3),reach[4,:num_runs],label='RLfD through shaping',color='y')
# plt.plot(range(50,80,3),reach[2,:num_runs],label='BQfD',color='r')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1))
# plt.tight_layout()
# plt.savefig('chain_rlfd')
