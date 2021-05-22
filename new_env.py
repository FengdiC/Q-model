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
import matplotlib.pyplot as plt


def find_vars(name):
    t_vars = tf.trainable_variables()
    f_vars = [var for var in t_vars if name in var.name]
    return f_vars



class mini_seaquest_dive:
    def __init__(self, env_type, init_pos=[0, 1, 2, 3, 4], rewards_value=[1, 2, 3, 4], rewards_indices=[6, 7, 8, 9], env_size=12, max_timestep=100):
        self.max_timestep = max_timestep
        self.env_size = env_size
        self.env_type = env_type
        self.rewards = np.zeros((env_size, ))
        self.init_pos = init_pos
        self.n_actions = 3

        self.rewards_value = rewards_value
        self.rewards_indices = rewards_indices
        #action, 0=up, 1=stay, 2=down
        #state, current_position/env_size, current_timestep/100, timestep until resurface

        for i in range(len(self.rewards_value)):
            self.rewards[self.rewards_indices[i]] = self.rewards_value[i]

    def reset(self):
        self.current_time = 0
        self.current_position = self.init_pos[np.random.randint(0, len(self.init_pos))]
        state = np.array([self.current_position/self.env_size, self.current_time/self.max_timestep]) * 2 - 1
        return state

    def step(self, action):
        self.current_time += 1
        if action == 0:
            #print("Ever here?", self.current_position)
            self.current_position = np.clip(self.current_position - 1, 0, self.env_size - 1)
            #print("Ever here?", self.current_position, "--------------------------------\n\n\n")
        elif action == 2:
            self.current_position = np.clip(self.current_position + 1, 0, self.env_size - 1)

        reward = self.rewards[int(self.current_position)]
        done = 0
        if self.current_time > self.max_timestep:
            done = 1
        if self.env_type == "dive":
            if self.current_time > 0 and self.current_time % 20 == 0 and not self.current_position == 0:
                done = 1
            elif self.current_time > 0 and self.current_time % 20 == 0 and self.current_position == 0:
                #print("Resurfaced ... ", done)
                reward += 0.0
        state = np.array([self.current_position/self.env_size, self.current_time/self.max_timestep]) * 2 - 1
        return state, reward, done

class DQN:
    """Implements a Deep Q Network"""
    def __init__(self, args, n_actions=3, hidden=256, input_size=2, agent_history_length=4, name="dqn", agent='dqn'):
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
        self.input_size = input_size
        self.var = args.var
        self.eta = args.eta
        self.agent = agent

        self.input = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
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
            d1 = tf.layers.dense(inputs=self.input, units=hidden, activation=tf.nn.relu, 
                                    kernel_initializer=tf.glorot_normal_initializer(), name="fc1")
            d2 = tf.layers.dense(inputs=d1, units=hidden, activation=tf.nn.relu, 
                                    kernel_initializer=tf.glorot_normal_initializer(), name="fc2")
            dense = tf.layers.dense(inputs=d2, units=hidden,activation=tf.nn.relu, 
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

def learn(session, states, actions, diffs, rewards, new_states, terminal_flags,
          weights, expert_idxes, main_dqn, target_dqn, batch_size, gamma, agent):
    states = np.squeeze(states)
    # print(states.shape)
    new_states = np.squeeze(new_states)
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input: new_states})
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input: new_states})
    double_q = q_vals[range(batch_size), arg_q_max]

    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = rewards + (gamma * double_q * (1 - terminal_flags))
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
            replay_buffer.add(obs_t=frame, reward=reward, action=action, done=terminal)
            if terminal:
                replay_buffer.add(obs_t=next_frame, reward=0, action=action_getter.get_random_action(), done=terminal)
            frame = next_frame
            frame_num += 1
            if terminal:
                break

def train_step_dqfd(sess, args, env, MAIN_DQN, TARGET_DQN, network_updater, replay_buffer, frame_num, eps_length,
                    learn, action_getter, grid, agent, pretrain=False):
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
    BS = args.batch_size
    terminal = False
    frame = env.reset()
    for _ in range(eps_length):
        if not pretrain:
            if args.stochastic_exploration == "True":
                action = action_getter.get_stochastic_action(sess, frame, MAIN_DQN)
            else:
                action = action_getter.get_action(sess, frame_num, frame, MAIN_DQN, evaluation=False, temporal=True)
            next_frame, reward, terminal = env.step(action)
            replay_buffer.add(obs_t=frame, reward=reward, action=action, done=terminal)
            if terminal:
                replay_buffer.add(obs_t=next_frame, reward=0, action=action_getter.get_random_action(), done=terminal)
            frame = next_frame
            episode_length += 1
            episode_reward_sum += reward
        frame_num += 1

        if (frame_num % UPDATE_FREQ == 0 and frame_num > grid - 1) or pretrain:
            generated_states, generated_actions, generated_diffs, generated_rewards, generated_new_states, \
            generated_terminal_flags, generated_weights, idxes, expert_idxes = replay_buffer.sample(
                BS, args.beta, expert=pretrain)  # Generated trajectories
            loss, loss_dq, loss_jeq = learn(sess, generated_states, generated_actions, generated_diffs,
                                            generated_rewards,
                                            generated_new_states, generated_terminal_flags, generated_weights,
                                            expert_idxes, MAIN_DQN, TARGET_DQN, BS, DISCOUNT_FACTOR, agent)
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
                #print("EPS terminated ... ", episode_length, episode_reward_sum, reward)
                break

    return episode_reward_sum, episode_length, np.mean(episode_loss), np.mean(episode_dq_loss), \
           np.mean(episode_jeq_loss), time.time() - start_time, np.mean(expert_ratio), regret_list, frame_list


def train(priority=True, agent='model', grid=10, seed=0):
    tf.reset_default_graph()
    with tf.variable_scope(agent):
        args = utils.argsparser()
        agent=args.agent
        tf.random.set_random_seed(args.seed)
        np.random.seed(args.seed)
        
        MAX_EPISODE_LENGTH = 100

        REPLAY_MEMORY_START_SIZE = args.replay_start_size  # Number of completely random actions,
        # before the agent starts learning
        MAX_FRAMES = 50000000  # Total number of frames the agent sees
        MEMORY_SIZE = args.replay_mem_size  # grid * grid +2 # Number of transitions stored in the replay memory
        # evaluation episode
        HIDDEN = 256
        BS = args.batch_size
        # main DQN and target DQN networks:
        print("Agent: ", agent)

        if args.env_id == "dive":
            env = mini_seaquest_dive(env_type=args.env_id, max_timestep=MAX_EPISODE_LENGTH,
                                     init_pos=[0, 1, 2, 3, 4], rewards_value=[1, 1.25, 1.5, 2], rewards_indices=[6, 7, 8, 9], env_size=12)
        elif args.env_id == "layer":
            env = mini_seaquest_dive(env_type=args.env_id, max_timestep=MAX_EPISODE_LENGTH,
                                     init_pos=[0, 1], rewards_value=[1, 1.25, 1.5, 2], rewards_indices=[3, 5, 7, 9], env_size=12)
        elif args.env_id == "middle":
            env = mini_seaquest_dive(env_type=args.env_id, max_timestep=MAX_EPISODE_LENGTH,
                                     init_pos=[0, 1, 3], rewards_value=[2, 2, 2], rewards_indices=[5, 6, 7], env_size=12)

        with tf.variable_scope('mainDQN'):
            MAIN_DQN = DQN(args, env.n_actions, HIDDEN, name="mainDQN", agent=agent)
        with tf.variable_scope('targetDQN'):
            TARGET_DQN = DQN(args, env.n_actions, HIDDEN, name="targetDQN", agent=agent)

        init = tf.global_variables_initializer()
        # MAIN_DQN_VARS = tf.trainable_variables(scope="mainDQN")
        # TARGET_DQN_VARS = tf.trainable_variables(scope="targetDQN")
        MAIN_DQN_VARS = find_vars("mainDQN")
        TARGET_DQN_VARS = find_vars("targetDQN")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        if priority:
            print("Priority", grid, grid * grid)
            my_replay_memory = PriorityBuffer.PrioritizedReplayBuffer(MEMORY_SIZE, args.alpha,frame_dtype=np.float32,
                                                                      state_shape=[2],
                                                                      agent_history_length=1,
                                                                      agent=agent, batch_size=BS)
        else:
            print("Not Priority")
            my_replay_memory = PriorityBuffer.ReplayBuffer(MEMORY_SIZE, state_shape=[2],frame_dtype=np.float32,
                                                           agent_history_length=1, agent=agent, batch_size=BS)
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

        print("Agent: ", agent)
        regret_list = []
        max_eps = 500
        last_eval = 0
        #my_replay_memory.load_expert_data(args.expert_dir+args.expert_file)
        # print("Beginning to pretrain")
        # train_step_dqfd(
        #     sess, args, env, MAIN_DQN, TARGET_DQN, network_updater, my_replay_memory, frame_number,
        #     args.pretrain_bc_iter, learn, action_getter, grid, agent, pretrain=True)
        # print("done pretraining ,test prioritized buffer")
        # print("buffer expert size: ", my_replay_memory.expert_idx)

        #pretrain evaluation
        eval_reward, eval_length, eval_states, eval_reward_list = eval(env, action_getter, sess,
                                                              MAIN_DQN,TARGET_DQN, eps_number, MAX_EPISODE_LENGTH)
        tflogger.log_scalar("Episode/Evaluation_reward",eval_reward, 0)
        tflogger.log_scalar("Episode/Evaluation_len", eval_length, 0)
        fig = "./" + args.log_dir + "/" + agent + "_" + args.env_id + "_lr_" + str(args.lr)+"_var_"+str(args.var) + "_seed_" + str(
        args.seed) + "_" + args.custom_id + "/" + tflogger.current_time + "/" + "result_" + str(int(eval_reward)) + "_frame_" + str(int(frame_number)) + ".png"
        x = np.arange(eval_length + 1)
        y = eval_states 
        plt.plot(x, y, label="Distance to bottom, s")
        plt.plot(x, eval_reward_list, label="R(s)")
        plt.xlabel("Timestep")
        plt.ylabel("Distance to bottom/Reward")
        plt.legend()
        plt.savefig(fig)
        plt.close()
        

        build_initial_replay_buffer(sess, env, my_replay_memory, action_getter, MAX_EPISODE_LENGTH,
                                    REPLAY_MEMORY_START_SIZE, args)

        while frame_number < MAX_FRAMES:
            eps_rw, eps_len, eps_loss, eps_dq_loss, eps_jeq_loss, eps_time, exp_ratio, _, _ = train_step_dqfd(
                sess, args, env, MAIN_DQN, TARGET_DQN, network_updater, my_replay_memory, frame_number,
                MAX_EPISODE_LENGTH, learn, action_getter, grid, agent, pretrain=False)
            frame_number += eps_len
            eps_number += 1
            last_eval += eps_len
            # print("GridSize", grid, "EPS: ", eps_number, "Mean Reward: ", eps_rw, "seed", args.seed,'EPS Length: ',eps_len,
            #       "Level: ",env.level)
            tflogger.log_scalar("Episode/Reward", eps_rw, frame_number)
            tflogger.log_scalar("Episode/Length", eps_len, frame_number)
            tflogger.log_scalar("Episode/Loss/Total", eps_loss, frame_number)

            tflogger.log_scalar("Episode/Loss/DQ", eps_dq_loss, frame_number)
            tflogger.log_scalar("Episode/Loss/JEQ", eps_jeq_loss, frame_number)
            tflogger.log_scalar("Episode/Expert Ratio", exp_ratio, frame_number)
            tflogger.log_scalar("Episode/Exploration", action_getter.get_eps(frame_number), frame_number)


            if last_eval > 20000:
                last_eval=0
                eval_reward, eval_length, eval_states, eval_reward_list = eval(env ,action_getter,sess,MAIN_DQN, TARGET_DQN, eps_number, MAX_EPISODE_LENGTH)
                tflogger.log_scalar("Episode/Evaluation_reward",eval_reward, frame_number)
                tflogger.log_scalar("Episode/Evaluation_len", eval_length, frame_number)
                #graph the results ... 
                #
                fig = "./" + args.log_dir + "/" + agent + "_" + args.env_id + "_lr_" + str(args.lr)+"_var_"+str(args.var) + "_seed_" + str(
                args.seed) + "_" + args.custom_id + "/" + tflogger.current_time + "/" + "result_" + str(int(eval_reward)) + "_frame_" + str(int(frame_number)) + ".png"

                x = np.arange(eval_length + 1)
                y = eval_states 
                plt.plot(x, y, label="Distance to bottom, s")
                plt.plot(x, eval_reward_list, label="R(s)")
                plt.xlabel("Timestep")
                plt.ylabel("Distance to bottom/Reward")
                plt.legend()
                plt.savefig(fig)
                plt.close()

                print("Eval Results", eval_reward, eval_length)


def eval(env, action_getter, sess, MAIN_DQN, TARGET_DQN, frame_num, eps_length):
    rewards = 0
    reward_list = []
    length = 0
    states = []
    next_frame = env.reset()
    states.append(env.env_size - ((next_frame[0] + 1)/2 * env.env_size))
    reward_list.append(0)
    for _ in range(eps_length):
        action = action_getter.get_action(sess, 0, next_frame, MAIN_DQN, evaluation=True, temporal=False)
        next_frame, reward, terminal = env.step(action)
        rewards += reward
        length += 1

        states.append(env.env_size - ((next_frame[0] + 1)/2 * env.env_size))
        print(length, env.current_position, ((next_frame[0] + 1)/2) * env.env_size, env.env_size - ((next_frame[0] + 1)/2 * env.env_size))
        reward_list.append(reward)
        if terminal:
            break
    return rewards, length, states, reward_list

train()