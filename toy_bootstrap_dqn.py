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

def find_vars(name):
    t_vars = tf.trainable_variables()
    f_vars = [var for var in t_vars if name in var.name]
    return f_vars

class DQN:
    """Implements a Deep Q Network"""
    def __init__(self, args, n_actions=2, hidden=512, grid=10, bootstrap_dqn_num=20, agent='dqfd', name="dqn"):
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

        self.input = tf.placeholder(shape=[None, self.grid * self.grid],dtype=tf.float32)
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
        self.loss, self.loss_per_sample = self.dqn_loss(MAIN_DQN_VARS)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        self.update = self.optimizer.minimize(self.loss, var_list=MAIN_DQN_VARS)

    def build_network(self, hidden, index=0):
        #layers
        with tf.variable_scope('Q_network_' + str(index)):
            layer1 = tf.layers.dense(inputs=self.input, units = hidden, activation=tf.nn.relu,
                                        kernel_initializer=tf.variance_scaling_initializer(scale=2),name='fc1')
            layer2 = tf.layers.dense(inputs=layer1, units=hidden, activation=tf.nn.relu,
                                        kernel_initializer=tf.variance_scaling_initializer(scale=2), name='fc2')
            dense = tf.layers.dense(inputs=layer2, units=hidden,
                                    kernel_initializer=tf.variance_scaling_initializer(scale=2), name="fc3")

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



    def expert_loss(self, t_vars):
        self.prob = tf.reduce_sum(tf.multiply(self.action_prob, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)),
                               axis=1)
        self.posterior = self.target_q+self.diff *(1-self.prob)
        l_dq = tf.losses.huber_loss(labels=self.posterior, predictions=self.Q, weights=self.weight*self.policy,
                                    reduction=tf.losses.Reduction.NONE)
        l_n_dq = tf.losses.huber_loss(labels=self.target_n_q, predictions=self.Q, weights=self.weight*self.policy,
                                      reduction=tf.losses.Reduction.NONE)

        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq
        self.l_n_dq = l_n_dq
        self.l_jeq = self.diff *(1-self.prob)/(self.target_q+0.001)

        loss_per_sample = l_dq
        loss = tf.reduce_mean(loss_per_sample+l2_reg_loss)
        return loss, loss_per_sample

    def dqfd_loss(self, t_vars):
        l_dq = tf.losses.huber_loss(labels=self.target_q, predictions=self.Q, weights=self.weight,
                                    reduction=tf.losses.Reduction.NONE)
        l_n_dq = tf.losses.huber_loss(labels=self.target_n_q, predictions=self.Q, weights=self.weight,
                                      reduction=tf.losses.Reduction.NONE)
        l_jeq = self.loss_jeq()

        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq
        self.l_n_dq = l_n_dq
        self.l_jeq = l_jeq

        loss_per_sample = self.l_dq + self.args.LAMBDA_1 * self.l_n_dq
        loss = tf.reduce_mean(loss_per_sample+self.l2_reg_loss+ self.args.LAMBDA_2 * self.l_jeq)
        return loss, loss_per_sample


    def dqn_loss(self,t_vars):
        l_dq = tf.losses.huber_loss(labels=self.target_q, predictions=self.Q, weights=self.weight, reduction=tf.losses.Reduction.NONE)
        l_n_dq = tf.losses.huber_loss(labels=self.target_n_q, predictions=self.Q, weights=self.weight, reduction=tf.losses.Reduction.NONE)
        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq
        self.l_n_dq = l_n_dq
        self.l_jeq = tf.constant(0)

        loss_per_sample = l_dq
        loss = tf.reduce_mean(loss_per_sample+l2_reg_loss)
        return loss, loss_per_sample


def learn(session, states, actions, diffs, rewards, new_states, terminal_flags,
            weights, expert_idxes, main_dqn, target_dqn, batch_size, gamma, args):
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
    new_states = np.squeeze(new_states)
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})
    double_q = q_vals[range(batch_size), arg_q_max]

    prob = session.run(target_dqn.action_prob, feed_dict={target_dqn.input:states})
    action_prob = prob[range(batch_size), actions]

    mask = np.where(diffs>0,np.ones((batch_size,)),np.zeros((batch_size,)))
    action_prob = mask * action_prob + (1-mask) * np.ones((batch_size,))
    action_prob = action_prob ** 0.2
    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = rewards + (gamma*double_q *  (1-terminal_flags))

    # arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:n_step_states})
    # q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:n_step_states})
    # double_q = q_vals[range(batch_size), arg_q_max]
    # target_n_q = n_step_rewards + (gamma*double_q * not_terminal)
    loss_sample, l_dq, l_jeq, _ = session.run([main_dqn.loss_per_sample, main_dqn.l_dq,
                                                main_dqn.l_jeq, main_dqn.update],
                          feed_dict={main_dqn.input:states,
                                     main_dqn.target_q:target_q,
                                     main_dqn.action:actions,
                                     main_dqn.expert_state:expert_idxes,
                                     main_dqn.weight:weights,
                                     main_dqn.diff: diffs,
                                     main_dqn.policy:action_prob
                                     })
    return loss_sample, np.mean(l_dq), np.mean(l_jeq)

class toy_env:
    def __init__(self, grid, final_reward=1, min_expert_frames=512, expert=True):
        self.grid = grid
        self.final_reward = final_reward
        self.reset()
        if expert:
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
        return self.get_current_state(), reward, terminal

    def generate_expert_data(self, min_expert_frames=512, expert_ratio=2):
        print("Creating Expert Data ... ")
        expert = {}
        half_expert_traj = self.grid//expert_ratio
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

def build_initial_replay_buffer(sess, env, replay_buffer, action_getter, max_eps, replay_buf_size, args):
    frame_num = 0
    while frame_num < replay_buf_size:
        _ = env.reset()
        for _ in range(max_eps):
            # 
            action = action_getter.get_random_action()
            # print("Action: ",action)
            # 
            next_frame, reward, terminal = env.step(action)
            #  Store transition in the replay memory
            replay_buffer.add(obs_t=next_frame, reward=reward, action=action, done=terminal)
            frame_num += 1
            if terminal:
                break

def train_step_dqfd(sess, args, env, bootstrap_dqns, replay_buffer, frame_num, eps_length,
                    learn, action_getter,grid, pretrain=False):
    start_time = time.time()
    episode_reward_sum = 0
    episode_length = 0
    episode_loss = []

    episode_dq_loss = []
    episode_jeq_loss = []

    expert_ratio = []

    NETW_UPDATE_FREQ = 2048        # Number of chosen actions between updating the target network.
    DISCOUNT_FACTOR = args.gamma           # gamma in the Bellman equation
    UPDATE_FREQ = args.update_freq                  # Every four actions a gradient descend step is performed
    BS = 8
    terminal = False
    frame = env.reset()
    #select a dqn ... 
    selected_dqn = bootstrap_dqns[np.random.randint(0, len(bootstrap_dqns))]
    priority = np.zeros((BS, ), dtype=np.float32)
    priority_weight = np.zeros((BS,), dtype=np.float32)
    for _ in range(eps_length):
        if args.stochastic_exploration == "True":
            action = action_getter.get_stochastic_action(sess, frame, selected_dqn["main"])
        else:
            action = action_getter.get_action(sess, frame_num, frame, selected_dqn["main"])
        next_frame, reward, terminal = env.step(action)
        replay_buffer.add(obs_t=next_frame, reward=reward, action=action, done=terminal)
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
                                                       expert_idxes, MAIN_DQN, TARGET_DQN, BS,DISCOUNT_FACTOR, args)
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


def train(priority=True, model_name='model', num_bootstrap=10):
    with tf.variable_scope(model_name):
        args = utils.argsparser()
        grid = args.grid_size
        name = args.agent
        tf.random.set_random_seed(args.seed)
        np.random.seed(args.seed)

        MAX_EPISODE_LENGTH = grid - 1
        EVAL_FREQUENCY = args.eval_freq  # Number of frames the agent sees between evaluations

        REPLAY_MEMORY_START_SIZE = 32 * 200 # Number of completely random actions,
        # before the agent starts learning
        MAX_FRAMES = 50000000  # Total number of frames the agent sees
        MEMORY_SIZE = 32 * 4000#grid * grid +2 # Number of transitions stored in the replay memory
        # evaluation episode
        HIDDEN = 256
        PRETRAIN = 8*grid
        BS = 8
        # main DQN and target DQN networks:
        print("Agent: ", name)

        bootstrap_dqns = []
        for i in range(num_bootstrap):
            print("Making network ", i)
            with tf.variable_scope('mainDQN_' + str(i)):
                MAIN_DQN = DQN(args, 2, HIDDEN, agent=name, grid=grid, name="mainDQN_" + str(i))
            with tf.variable_scope('targetDQN_' + str(i)):
                TARGET_DQN = DQN(args, 2, HIDDEN, agent=name, grid=grid, name="targetDQN_" + str(i))
            MAIN_DQN_VARS = find_vars("mainDQN_" + str(i))
            TARGET_DQN_VARS = find_vars("targetDQN_" + str(i))
            network_updater = utils.TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
            bootstrap_dqns.append({"main":MAIN_DQN, "target":TARGET_DQN, "updater":network_updater, "idx":i})

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if priority:
            print("Priority", grid, grid * grid)
            my_replay_memory = PriorityBuffer.PrioritizedReplayBuffer(MEMORY_SIZE, args.alpha, state_shape=[grid * grid], agent_history_length=1, agent=name, batch_size=BS, bootstrap=num_bootstrap)
        else:
            print("Not Priority")
            my_replay_memory = PriorityBuffer.ReplayBuffer(MEMORY_SIZE, state_shape=[grid * grid],agent_history_length=1, agent=name, batch_size=BS, bootstrap=num_bootstrap)
        action_getter = utils.ActionGetter(2,
                                        replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                        max_frames=MAX_FRAMES,
                                        eps_initial=args.initial_exploration)
        saver = tf.train.Saver(max_to_keep=10)
        sess = tf.Session(config=config)
        sess.run(init)
        # saver.restore(sess, "../models/" + name + "/" + args.env_id + "/"  + "model-" + str(5614555))

        eps_number = 0
        frame_number = grid

        tflogger = tensorflowboard_logger(
            "./" + args.log_dir + "/" + "toy_bootstrap_"+"/"+name + "_" + args.env_id + "_seed_" + str(args.seed),
            sess, args)

        env = toy_env(grid, expert=False)
        print("Agent: ", name)
        frame_num = 0
        print_iter = 25
        last_eval = 0
        initial_time = time.time()
        # eval_rewards, eval_len, eval_pos = eval_env(sess, args, env, MAIN_DQN, TARGET_DQN, network_updater, my_replay_memory,  frame_number,
        # MAX_EPISODE_LENGTH, learn, action_getter, grid, pretrain=False)
        build_initial_replay_buffer(sess, env, my_replay_memory, action_getter, MAX_EPISODE_LENGTH, REPLAY_MEMORY_START_SIZE, args)
        # tflogger.log_scalar("Evaluation/Reward", eval_rewards, eps_number)
        # tflogger.log_scalar("Evaluation/Len", eval_len, eps_number)
        # for i in range(len(eval_pos)):
        #     print(i, eval_pos[i])
        min_eps = 100
        reward_list = []
        while frame_number < MAX_FRAMES:
            eps_rw, eps_len, eps_loss, eps_dq_loss, eps_jeq_loss, eps_time, exp_ratio = train_step_dqfd(
                sess, args, env, bootstrap_dqns, my_replay_memory,  frame_number,
                MAX_EPISODE_LENGTH, learn, action_getter, grid, pretrain=False)
            frame_number += eps_len
            eps_number += 1
            last_eval += eps_len

            if eps_number % print_iter == 0:
                if my_replay_memory.expert_idx > my_replay_memory.agent_history_length:
                    tflogger.log_scalar("Expert Priorities",
                                        my_replay_memory._it_sum.sum(my_replay_memory.agent_history_length,
                                                                    my_replay_memory.expert_idx), eps_number)
                    tflogger.log_scalar("Total Priorities",
                                        my_replay_memory._it_sum.sum(my_replay_memory.agent_history_length,
                                                                    my_replay_memory.count - 1), eps_number)
            tflogger.log_scalar("Episode/Reward", eps_rw, eps_number)
            tflogger.log_scalar("Episode/Length", eps_len, eps_number)
            tflogger.log_scalar("Episode/Loss/Total", eps_loss, eps_number)
            tflogger.log_scalar("Episode/Time", eps_time, eps_number)
            tflogger.log_scalar("Episode/Reward Per Frame", eps_rw / max(1, eps_len),
                                eps_number)
            tflogger.log_scalar("Episode/Expert Ratio", exp_ratio, eps_number)
            tflogger.log_scalar("Episode/Exploration", action_getter.get_eps(frame_number), eps_number)
            tflogger.log_scalar("Episode/Loss/DQ", eps_dq_loss, eps_number)
            tflogger.log_scalar("Episode/Loss/JEQ", eps_jeq_loss, eps_number)
            tflogger.log_scalar("Episode/Time", eps_time, eps_number)
            tflogger.log_scalar("Episode/Reward/Reward Per Frame", eps_rw / max(1, eps_len),
                                eps_number)
            tflogger.log_scalar("Episode/Expert Ratio", exp_ratio, eps_number)
            tflogger.log_scalar("Episode/Exploration", action_getter.get_eps(frame_number), eps_number)

            tflogger.log_scalar("Total Episodes", eps_number, frame_number)
            tflogger.log_scalar("Replay Buffer Size", my_replay_memory.count, eps_number)
            tflogger.log_scalar("Elapsed Time", time.time() - initial_time, eps_number)
            tflogger.log_scalar("Frames Per Hour", frame_number / ((time.time() - initial_time) / 3600), eps_number)

            current_list = []
            for index in range(len(bootstrap_dqns)):
                eps_rw, _, _ = eval_env(sess, args, env, bootstrap_dqns, frame_number, MAX_EPISODE_LENGTH, action_getter, grid, index=index)
                current_list.append(eps_rw)
            print(eps_number, np.mean(current_list), np.mean(eps_loss), frame_number)
            if np.mean(current_list) > 0.1:
                return frame_number


            # reward_list.append(eps_rw)
            # if eps_len % 100:
            #     print(eps_len, np.mean(reward_list))
            # if eps_rw > 0.5:
            #     print("GridSize", grid, "EPS: ", eps_number, "Mean Reward: ", np.mean(reward_list[-20:]), "seed", args.seed, frame_number)
            #     return frame_number, np.mean(reward_list)
            # if eps_number % 1000:
            #     eval_rewards, eval_len, eval_pos = eval_env(sess, args, env, MAIN_DQN, TARGET_DQN, network_updater, 
            #                                                 my_replay_memory,  frame_number, MAX_EPISODE_LENGTH, learn, 
            #                                                 action_getter, grid, pretrain=False)

train()

