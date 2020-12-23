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


def compute_regret(Q_value, grid, gamma, final_reward=1):
    # pi = np.zeros((grid, grid), dtype=np.uint8)
    # for i in range(grid):
    #     pi[i, i] = 1

    pi = np.argmax(Q_value, axis=2)
    P = np.zeros((grid * grid, grid * grid))
    R = np.zeros((grid, grid))
    for i in range(grid - 1):
        for j in range(grid - 1):
            action = int(pi[i, j])
            l = i + 1
            if action == 0:
                r = max(0, j - 1)
                R[i, j] = 0
            else:
                r = min(j + 1, grid - 1)
                if r == grid - 1:
                    R[i, j] = final_reward
                else:
                    R[i, j] = -0.01 / grid
            P[i * grid + j, l * grid + r] = 1
    for j in range(grid - 1):
        P[(grid - 1) * grid + j, (grid - 1) * grid + j] = 1
    R = np.ndarray.flatten(R)
    Q = np.matmul(np.linalg.inv(np.eye(grid * grid) - gamma * P), R)
    return Q[0]


class DQN:
    """Implements a Deep Q Network"""

    def __init__(self, args, n_actions=4, hidden=256, grid=10, agent_history_length=6, name="dqn", agent='dqn'):
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
        self.agent = agent

        self.input = tf.placeholder(shape=[None, self.grid,self.grid,agent_history_length], dtype=tf.float32)
        self.weight = tf.placeholder(shape=[None, ], dtype=tf.float32)
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
        if self.agent == 'bootdqn':
            self.loss, self.loss_per_sample = self.dqn_loss(MAIN_DQN_VARS)
        elif self.agent == 'expert':
            self.loss, self.loss_per_sample = self.expert_loss(MAIN_DQN_VARS)
        elif self.agent == 'dqfd':
            self.loss, self.loss_per_sample = self.dqfd_loss(MAIN_DQN_VARS)
        else:
            self.loss, self.loss_per_sample = self.dqn_loss(MAIN_DQN_VARS)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        self.update = self.optimizer.minimize(self.loss, var_list=MAIN_DQN_VARS)

    def build_network(self, hidden, index=0):
        # layers
        with tf.variable_scope('Q_network_' + str(index)):
            conv1 = tf.layers.conv2d(
                inputs=self.input, filters=32, kernel_size=[1,1], strides=1,
                kernel_initializer=tf.glorot_normal_initializer(),
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1')
            conv2 = tf.layers.conv2d(
                inputs=conv1, filters=64, kernel_size=[1,1], strides=1,
                kernel_initializer=tf.glorot_normal_initializer(),
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2')
            d = tf.layers.flatten(conv2)
            dense = tf.layers.dense(inputs=d, units=hidden,
                                    kernel_initializer=tf.glorot_normal_initializer(), name="fc3")

            # Splitting into value and advantage stream
            valuestream, advantagestream = tf.split(dense, 2, -1)
            valuestream = tf.layers.flatten(valuestream)
            advantagestream = tf.layers.flatten(advantagestream)
            advantage = tf.layers.dense(
                inputs=advantagestream, units=self.n_actions,
                kernel_initializer=tf.glorot_normal_initializer(), name="advantage")
            value = tf.layers.dense(
                inputs=valuestream, units=1,
                kernel_initializer=tf.glorot_normal_initializer(), name='value')
            q_values = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))
            return q_values

    def loss_jeq(self):
        expert_act_one_hot = tf.one_hot(self.action, self.n_actions, dtype=tf.float32)
        self.expert_q_value = tf.reduce_sum(expert_act_one_hot * self.q_values, axis=1)
        # JEQ = max_{a in A}(Q(s,a) + l(a_e, a)) - Q(s, a_e)
        self.q_plus_margin = self.q_values + (1 - expert_act_one_hot) * self.args.dqfd_margin
        self.max_q_plus_margin = tf.reduce_max(self.q_plus_margin, axis=1)
        jeq = (self.max_q_plus_margin - self.expert_q_value) * self.expert_state
        return jeq

    def expert_loss(self, t_vars):
        ratio = (16 + 4 * self.diff) / tf.square(4 + self.diff)

        self.prob = tf.reduce_sum(
            tf.multiply(self.action_prob, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)),
            axis=1)
        self.posterior = self.target_q + self.eta * self.var * ratio * (1 - self.prob) * self.expert_state
        l_dq = tf.losses.huber_loss(labels=self.posterior, predictions=self.Q, weights=self.weight,
                                    reduction=tf.losses.Reduction.NONE)

        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq 
        self.l_jeq = self.diff * (1 - self.prob) / (self.target_q + 0.001)

        loss_per_sample = l_dq
        loss = tf.reduce_mean(loss_per_sample + l2_reg_loss)
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
        loss = tf.reduce_mean(loss_per_sample + self.l2_reg_loss + self.args.LAMBDA_2 * self.l_jeq)
        return loss, loss_per_sample

    def dqn_loss(self, t_vars):
        l_dq = tf.losses.huber_loss(labels=self.target_q, predictions=self.Q, weights=self.weight,
                                    reduction=tf.losses.Reduction.NONE)
        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq
        self.l_jeq = tf.constant(0)

        loss_per_sample = l_dq
        loss = tf.reduce_mean(loss_per_sample + l2_reg_loss)
        return loss, loss_per_sample

    def get_q_value(self, sess):
        q_values = np.zeros((self.grid, self.grid, self.n_actions), dtype=np.float32)
        for i in range(self.grid):
            for j in range(self.grid):
                state = np.array([i + 1, j + 1])
                state = np.reshape(state, (1, 2))
                value = sess.run(self.q_values, feed_dict={self.input: state/(0.5*grid)-1})
                q_values[i, j] = value[0]
        return q_values


def learn(session, states, actions, diffs, rewards, new_states, terminal_flags,
          weights, expert_idxes, main_dqn, target_dqn, batch_size, gamma, agent, shaping=None):
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
    states = np.squeeze(states)
    # print(states.shape)
    new_states = np.squeeze(new_states)
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input: new_states})
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input: new_states})
    double_q = q_vals[range(batch_size), arg_q_max]

    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = rewards + (gamma * double_q * (1 - terminal_flags))
    if agent == 'shaping':
        state_indices = np.round(((states+1)*0.5 * main_dqn.grid)).astype(np.uint8)
        new_state_indices = np.round(((new_states+1)*0.5 * main_dqn.grid)).astype(np.uint8)
        current_potential = shaping[state_indices[:,0].astype(np.uint8),state_indices[:,1].astype(np.uint8)]
        next_potential = shaping[new_state_indices[:,0].astype(np.uint8),new_state_indices[:,1].astype(np.uint8)]
        curr = current_potential[range(batch_size), actions]
        next = next_potential[range(batch_size), arg_q_max]
        target_q += gamma * next * (1 - terminal_flags) - curr

    loss_sample, l_dq, l_jeq, _ = session.run([main_dqn.loss_per_sample, main_dqn.l_dq,
                                               main_dqn.l_jeq, main_dqn.update],
                                              feed_dict={main_dqn.input: states,
                                                         main_dqn.target_q: target_q,
                                                         main_dqn.action: actions,
                                                         main_dqn.expert_state: expert_idxes,
                                                         main_dqn.weight: weights,
                                                         main_dqn.diff: diffs
                                                         })
    return loss_sample, np.mean(l_dq), np.mean(l_jeq)


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
            replay_buffer.add(obs_t=frame[:,:,-1], reward=reward, action=action, done=terminal)
            frame = next_frame
            frame_num += 1
            if terminal:
                break


def train_step_bootdqn(sess, args, env, bootstrap_dqns, replay_buffer, frame_num, eps_length,
                       learn, action_getter, grid, agent, pretrain=False):
    start_time = time.time()
    episode_reward_sum = 0
    episode_length = 0
    episode_loss = []

    episode_dq_loss = []
    episode_jeq_loss = []

    expert_ratio = []

    NETW_UPDATE_FREQ = 25 * grid  # Number of chosen actions between updating the target network.
    DISCOUNT_FACTOR = args.gamma  # gamma in the Bellman equation
    UPDATE_FREQ = args.update_freq  # Every four actions a gradient descend step is performed
    BS = 32
    terminal = False
    # select a dqn ...
    selected_dqn = bootstrap_dqns[np.random.randint(0, len(bootstrap_dqns))]
    priority = np.zeros((BS,), dtype=np.float32)
    priority_weight = np.zeros((BS,), dtype=np.float32)
    frame = env.reset()
    for _ in range(eps_length):
        if not pretrain:
            if args.stochastic_exploration == "True":
                action = action_getter.get_stochastic_action(sess, frame, selected_dqn["main"])
            else:
                action = action_getter.get_action(sess, frame_num, frame, selected_dqn["main"], evaluation=True)
            next_frame, reward, terminal = env.step(action)
            replay_buffer.add(obs_t=frame[:,:,-2:], reward=reward, action=action, done=terminal)
            frame = next_frame
            episode_length += 1
            episode_reward_sum += reward
        frame_num += 1

        if frame_num % UPDATE_FREQ == 0 and frame_num > grid - 1:
            generated_states, generated_actions, generated_diffs, generated_rewards, generated_new_states, \
            generated_terminal_flags, generated_weights, idxes, expert_idxes = replay_buffer.sample(
                BS, args.beta, expert=pretrain)  # Generated trajectories
            bootstrap_mask = replay_buffer.get_bootstrap(idxes)

            for bootstrap_dqn in bootstrap_dqns:
                MAIN_DQN = bootstrap_dqn["main"]
                TARGET_DQN = bootstrap_dqn["target"]
                mask = bootstrap_mask[:, bootstrap_dqn["idx"]]
                generated_weights = generated_weights * mask

                loss, loss_dq, loss_jeq = learn(sess, generated_states, generated_actions, generated_diffs,
                                                generated_rewards,
                                                generated_new_states, generated_terminal_flags, generated_weights,
                                                expert_idxes, MAIN_DQN, TARGET_DQN, BS, DISCOUNT_FACTOR, agent)
                episode_dq_loss.append(loss_dq)
                episode_jeq_loss.append(loss_jeq)

                priority += loss
                priority_weight += mask
            priority_weight = np.clip(priority_weight, 1, BS * 2)
            priority = priority / priority_weight
            replay_buffer.update_priorities(idxes, priority, expert_idxes, frame_num,
                                            expert_priority_modifier=args.expert_priority_modifier,
                                            min_expert_priority=args.min_expert_priority, pretrain=pretrain)
            expert_ratio.append(np.sum(expert_idxes) / BS)
            episode_loss.append(loss)
        if pretrain:
            if frame_num % (NETW_UPDATE_FREQ // UPDATE_FREQ) == 0 and frame_num > 0:
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
            if env.terminal:
                env.reset()
    return episode_reward_sum, episode_length, np.mean(episode_loss), np.mean(episode_dq_loss), \
           np.mean(episode_jeq_loss), time.time() - start_time, np.mean(expert_ratio)


def train_bootdqn(priority=True, agent='model', num_bootstrap=10, seed=0, grid=10):
    tf.reset_default_graph()
    with tf.variable_scope(agent):
        args = utils.argsparser()
        agent = args.agent
        tf.random.set_random_seed(args.seed)
        np.random.seed(args.seed)
        NETW_UPDATE_FREQ = 25 * grid  # Number of chosen actions between updating the target network.

        MAX_EPISODE_LENGTH = grid * grid

        REPLAY_MEMORY_START_SIZE = 32 * 200  # Number of completely random actions,
        # before the agent starts learning
        MAX_FRAMES = 50000000  # Total number of frames the agent sees
        MEMORY_SIZE = 32 * 4000  # grid * grid +2 # Number of transitions stored in the replay memory
        # evaluation episode
        HIDDEN = 512
        BS = 32
        # main DQN and target DQN networks:
        print("Agent: ", agent)

        if args.env_id == 'maze':
            env = toy_maze('mazes')
        else:
            final_reward = 1
            env = toy_env(grid, final_reward)

        bootstrap_dqns = []
        for i in range(num_bootstrap):
            print("Making network ", i)
            with tf.variable_scope('mainDQN_' + str(i)):
                MAIN_DQN = DQN(args, env.n_actions, HIDDEN, grid=grid, name="mainDQN_" + str(i), agent=agent)
            with tf.variable_scope('targetDQN_' + str(i)):
                TARGET_DQN = DQN(args, env.n_actions, HIDDEN, grid=grid, name="targetDQN_" + str(i), agent=agent)
            MAIN_DQN_VARS = find_vars("mainDQN_" + str(i))
            TARGET_DQN_VARS = find_vars("targetDQN_" + str(i))
            network_updater = utils.TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
            bootstrap_dqns.append({"main": MAIN_DQN, "target": TARGET_DQN, "updater": network_updater, "idx": i})

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if priority:
            print("Priority", grid, grid * grid)
            my_replay_memory = PriorityBuffer.PrioritizedReplayBuffer(MEMORY_SIZE, args.alpha,frame_dtype=np.float32,
                                                                      state_shape=[args.state_size],
                                                                      agent_history_length=1, agent=agent, batch_size=BS,
                                                                      bootstrap=num_bootstrap)
        else:
            print("Not Priority")
            my_replay_memory = PriorityBuffer.ReplayBuffer(MEMORY_SIZE, state_shape=[args.state_size],frame_dtype=np.float32,
                                                           agent_history_length=1, agent=agent, batch_size=BS,
                                                           bootstrap=num_bootstrap)
        action_getter = utils.ActionGetter(env.n_actions,
                                           replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                           max_frames=MAX_FRAMES,
                                           eps_initial=args.initial_exploration)
        saver = tf.train.Saver(max_to_keep=10)
        sess = tf.Session(config=config)
        sess.run(init)
        eps_number = 0
        frame_number = 0

        print("Agent: ", agent)
        last_eval = 0
        build_initial_replay_buffer(sess, env, my_replay_memory, action_getter, MAX_EPISODE_LENGTH,
                                    REPLAY_MEMORY_START_SIZE, args)
        max_eps = 500
        regret_list = []

        # compute regret
        # Q_value = np.zeros((grid, grid, 2))
        # Q_value[:, :, 1] = final_reward
        # V = compute_regret(Q_value, grid, args.gamma,final_reward)
        while frame_number < MAX_FRAMES:
            eps_rw, eps_len, eps_loss, eps_dq_loss, eps_jeq_loss, eps_time, exp_ratio = train_step_bootdqn(
                sess, args, env, bootstrap_dqns, my_replay_memory, frame_number,
                MAX_EPISODE_LENGTH, learn, action_getter, grid, agent, pretrain=False)
            frame_number += eps_len
            eps_number += 1
            last_eval += eps_len
            print("GridSize", grid, "EPS: ", eps_number, "Mean Reward: ", eps_rw, "seed", args.seed)

            if args.env_id == 'chain':
                q_values = MAIN_DQN.get_q_value(sess)
                pi = np.argmax(q_values, axis=2)
                correct = grid - 1 - np.sum(np.diag(pi)[:-1])
                # correct = np.sum(pi[:, 0])
                print(grid, eps_number, correct, eps_rw)
                regret_list.append(correct)
                # #compute regret
                # regret_list.append(V - compute_regret(q_values, grid, args.gamma, final_reward))
                # print(V, eps_number, regret_list[-1], eps_rw)
                if (len(regret_list) > 5 and np.mean(regret_list[-3:]) < 0.02) or eps_number > max_eps:
                    print("GridSize", grid, "EPS: ", eps_number, "Mean Reward: ", eps_rw, "seed", args.seed)
                    return eps_number


def potential(grid):
    potential = np.zeros((grid, grid, 2))
    for j in reversed(range(grid + 1)):
        for i in range(j):
            potential[i, i + grid - j, 1] = np.exp(-0.5 * (grid - j) ** 2)
            potential[i + grid - j, i, 1] = np.exp(-0.5 * (grid - j) ** 2)
    # potential = np.reshape(potential,(grid*grid,2))
    return potential


def train_step_dqfd(sess, args, env, MAIN_DQN, TARGET_DQN, network_updater, replay_buffer, frame_num, eps_length,
                    learn, action_getter, grid, shaping, agent, pretrain=False):
    start_time = time.time()
    episode_reward_sum = 0
    episode_length = 0
    episode_loss = []
    regret_list = []
    frame_list = []

    episode_dq_loss = []
    episode_jeq_loss = []

    expert_ratio = []

    NETW_UPDATE_FREQ = args.target_update_freq  # Number of chosen actions between updating the target network.
    DISCOUNT_FACTOR = args.gamma  # gamma in the Bellman equation
    UPDATE_FREQ = args.update_freq  # Every four actions a gradient descend step is performed
    BS = 32
    terminal = False
    frame = env.reset()
    for _ in range(eps_length):
        if not pretrain:
            if args.stochastic_exploration == "True":
                action = action_getter.get_stochastic_action(sess, frame, MAIN_DQN)
            elif agent != 'dqn':
                action = action_getter.get_action(sess, frame_num, frame, MAIN_DQN, evaluation=False, temporal=False)
            else:
                action = action_getter.get_action(sess, frame_num, frame, MAIN_DQN, evaluation=False, temporal=False)
            next_frame, reward, terminal = env.step(action)
            replay_buffer.add(obs_t=frame[:,:,-1], reward=reward, action=action, done=terminal)
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
                                            expert_idxes, MAIN_DQN, TARGET_DQN, BS, DISCOUNT_FACTOR, agent, shaping)
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
            if env.terminal:
                env.reset()
    return episode_reward_sum, episode_length, np.mean(episode_loss), np.mean(episode_dq_loss), \
           np.mean(episode_jeq_loss), time.time() - start_time, np.mean(expert_ratio), regret_list, frame_list


def potential_pretrain(session, states, actions, diffs, rewards, new_states, terminal_flags,
                       weights, expert_idxes, main_dqn, target_dqn, batch_size, gamma, agent, shaping=None):
    states = np.squeeze(states)
    state_indices = np.round(((states+1) * main_dqn.grid*0.5)).astype(np.uint8)
    current_potential = shaping[state_indices[:, 0].astype(np.uint8), state_indices[:, 1].astype(np.uint8)]
    target_q = current_potential[range(batch_size), actions]

    loss_sample, l_dq, l_jeq, _ = session.run([main_dqn.loss_per_sample, main_dqn.l_dq,
                                               main_dqn.l_jeq, main_dqn.update],
                                              feed_dict={main_dqn.input: states,
                                                         main_dqn.target_q: target_q,
                                                         main_dqn.action: actions,
                                                         main_dqn.expert_state: expert_idxes,
                                                         main_dqn.weight: weights,
                                                         main_dqn.diff: diffs
                                                         })
    return loss_sample, np.mean(l_dq), np.mean(l_jeq)


def train(priority=True, agent='model', grid=10, seed=0):
    tf.reset_default_graph()
    with tf.variable_scope(agent):
        args = utils.argsparser()
        agent=args.agent
        tf.random.set_random_seed(args.seed)
        np.random.seed(args.seed)
        shaping = potential(grid)

        MAX_EPISODE_LENGTH = 2*grid * grid

        REPLAY_MEMORY_START_SIZE = args.replay_start_size  # Number of completely random actions,
        # before the agent starts learning
        MAX_FRAMES = 50000000  # Total number of frames the agent sees
        MEMORY_SIZE = args.replay_mem_size  # grid * grid +2 # Number of transitions stored in the replay memory
        # evaluation episode
        HIDDEN = 512
        BS = 32
        # main DQN and target DQN networks:
        print("Agent: ", agent)

        if args.env_id == 'maze':
            env = toy_maze('mazes_large',expert_dir=args.expert_dir,level=10)
            env_val = toy_maze('test_mazes',expert_dir=args.expert_dir,level=5,expert=False)
            env_test = toy_maze('test_mazes_2',expert_dir=args.expert_dir,level=5,expert=False)

        with tf.variable_scope('mainDQN'):
            MAIN_DQN = DQN(args, env.n_actions, HIDDEN, grid=grid, name="mainDQN", agent=agent)
        with tf.variable_scope('targetDQN'):
            TARGET_DQN = DQN(args, env.n_actions, HIDDEN, grid=grid, name="targetDQN", agent=agent)

        init = tf.global_variables_initializer()
        MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
        TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        if priority:
            print("Priority", grid, grid * grid)
            my_replay_memory = PriorityBuffer.PrioritizedReplayBuffer(MEMORY_SIZE, args.alpha,frame_dtype=np.float32,
                                                                      state_shape=[grid,grid],
                                                                      agent_history_length=6,
                                                                      agent=agent, batch_size=BS)
        else:
            print("Not Priority")
            my_replay_memory = PriorityBuffer.ReplayBuffer(MEMORY_SIZE, state_shape=[grid,grid],frame_dtype=np.float32,
                                                           agent_history_length=6, agent=agent, batch_size=BS)
        network_updater = utils.TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
        action_getter = utils.ActionGetter(env.n_actions,eps_annealing_frames=MEMORY_SIZE, eps_final=0.05,
                                               replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                               max_frames=MAX_FRAMES,
                                               eps_initial=args.initial_exploration)
        saver = tf.train.Saver(max_to_keep=10)
        sess = tf.Session(config=config)
        sess.run(init)
        eps_number = 0
        frame_number = 0
        tflogger = tensorflowboard_logger(
            "./" + args.log_dir + "/" + agent + "_" + args.env_id + "_lr_" + str(args.lr)+"_var_"+str(args.var) + "_seed_" + str(
                args.seed) + "_" + args.custom_id, sess, args)

        my_replay_memory.load_expert_data(args.expert_dir+args.expert_file)

        print("Agent: ", agent)
        regret_list = []
        max_eps = 500
        # # compute regret
        # Q_value = np.zeros((grid, grid, 2))
        # Q_value[:, :, 1] = final_reward
        # V = compute_regret(Q_value, grid, args.gamma, final_reward)
        # print("True value for initial state: ", V)

        last_eval = 0
        if agent == 'shaping':
            print("Beginning to pretrain")
            train_step_dqfd(
                sess, args, env, MAIN_DQN, TARGET_DQN, network_updater, my_replay_memory, frame_number,
                args.pretrain_bc_iter, potential_pretrain, action_getter, grid, shaping, agent, pretrain=True)
            print("done pretraining ,test prioritized buffer")
            print("buffer expert size: ", my_replay_memory.expert_idx)
        else:
            print("Beginning to pretrain")
            train_step_dqfd(
                sess, args, env, MAIN_DQN, TARGET_DQN, network_updater, my_replay_memory, frame_number,
                args.pretrain_bc_iter, learn, action_getter, grid, shaping, agent, pretrain=True)
            print("done pretraining ,test prioritized buffer")
            print("buffer expert size: ", my_replay_memory.expert_idx)

        #pretrain evaluation
        test_eps_reward, val_reward, test_eps_reward_t = eval(args, env_test, env_val, env, action_getter, sess,
                                                              MAIN_DQN)
        print(test_eps_reward, val_reward, test_eps_reward_t)
        tflogger.log_scalar("Episode/Evaluation", test_eps_reward, frame_number)
        tflogger.log_scalar("Episode/Evaluation_Val", val_reward, frame_number)
        tflogger.log_scalar("Episode/Evaluation_Test", test_eps_reward_t, frame_number)

        build_initial_replay_buffer(sess, env, my_replay_memory, action_getter, MAX_EPISODE_LENGTH,
                                    REPLAY_MEMORY_START_SIZE, args)

        while frame_number < MAX_FRAMES:
            eps_rw, eps_len, eps_loss, eps_dq_loss, eps_jeq_loss, eps_time, exp_ratio, _, _ = train_step_dqfd(
                sess, args, env, MAIN_DQN, TARGET_DQN, network_updater, my_replay_memory, frame_number,
                MAX_EPISODE_LENGTH, learn, action_getter, grid, shaping, agent, pretrain=False)
            frame_number += eps_len
            eps_number += 1
            last_eval += 1
            # print("GridSize", grid, "EPS: ", eps_number, "Mean Reward: ", eps_rw, "seed", args.seed,'EPS Length: ',eps_len,
            #       "Level: ",env.level)
            tflogger.log_scalar("Episode/Reward", eps_rw, frame_number)
            tflogger.log_scalar("Episode/Length", eps_len, frame_number)
            tflogger.log_scalar("Episode/Level", env.level, frame_number)
            tflogger.log_scalar("Episode/Loss/Total", eps_loss, frame_number)

            tflogger.log_scalar("Episode/Loss/DQ", eps_dq_loss, frame_number)
            tflogger.log_scalar("Episode/Loss/JEQ", eps_jeq_loss, frame_number)
            tflogger.log_scalar("Episode/Expert Ratio", exp_ratio, frame_number)
            tflogger.log_scalar("Episode/Exploration", action_getter.get_eps(frame_number), frame_number)

            if args.env_id == 'chain':
                q_values = MAIN_DQN.get_q_value(sess)
                pi = np.argmax(q_values, axis=2)
                # print(np.diag(pi)[:-1])
                correct = grid - 1 - np.sum(np.diag(pi)[:-1])
                # correct = np.sum(pi[:,0])
                print(grid, eps_number, correct, eps_rw)
                regret_list.append(correct)

                # compute regret
                # regret_list.append(V - compute_regret(q_values, grid, args.gamma,  final_reward))
                # print(V, eps_number, regret_list[-1], eps_rw)
                # regret_list.append(eps_rw)
                if (len(regret_list) > 5 and np.mean(regret_list[-3:]) < 0.02 and env.final) or eps_number > max_eps:
                    print("GridSize", grid, "EPS: ", eps_number, "Mean Reward: ", eps_rw, "seed", args.seed)
                    return eps_number
            if last_eval > 100:
                last_eval=0
                test_eps_reward,val_reward,test_eps_reward_t = eval(args,env_test,env_val,env,action_getter,sess,MAIN_DQN)
                print(test_eps_reward,val_reward,test_eps_reward_t)
                tflogger.log_scalar("Episode/Evaluation",test_eps_reward, frame_number)
                tflogger.log_scalar("Episode/Evaluation_Val", val_reward, frame_number)
                tflogger.log_scalar("Episode/Evaluation_Test",test_eps_reward_t, frame_number)

def eval(args,env_test,env_val,env,action_getter,sess,MAIN_DQN):
    episode_length=0
    eps_reward=0
    env.restart()
    plot=False
    for level in range(15):
        terminal=False
        frame = env.reset(eval=True)
        episode_length = 0
        while episode_length < 80 and not terminal:
            action = action_getter.get_action(sess, 0, frame, MAIN_DQN, evaluation=True, temporal=False)
            if plot:
                plot_state(frame[:,:,-1])
                if episode_length>20:
                    plot = False
            next_frame, reward, terminal = env.step(action)
            frame = next_frame
            episode_length += 1
            eps_reward += reward
        plot=False
    val_eps_reward = 0
    env_val.restart()
    for level in range(5):
        terminal = False
        frame = env_val.reset(eval=True)
        episode_length = 0
        while episode_length < 80 and not terminal:
            action = action_getter.get_action(sess, 0, frame, MAIN_DQN, evaluation=True, temporal=False)
            next_frame, reward, terminal = env_val.step(action)
            frame = next_frame
            episode_length += 1
            val_eps_reward += reward
    test_eps_reward=0
    env_test.restart()
    for level in range(5):
        terminal=False
        frame = env_test.reset(eval=True)
        episode_length = 0
        while episode_length < 80 and not terminal:
            action = action_getter.get_action(sess, 0, frame, MAIN_DQN, evaluation=True, temporal=False)
            next_frame, reward, terminal = env_test.step(action)
            frame = next_frame
            episode_length += 1
            test_eps_reward += reward
    return eps_reward,val_eps_reward,test_eps_reward

import matplotlib.pyplot as plt
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


train()