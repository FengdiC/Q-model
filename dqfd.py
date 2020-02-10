import tensorflow as tf
import numpy as np
import os
import random
import math
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')
import gym
import imageio
from skimage.transform import resize
import time
import pickle
import utils
import PriorityBuffer

def softargmax(x, beta=1e10):
  x = tf.convert_to_tensor(x)
  x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
  return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)

class DQN:
    """Implements a Deep Q Network"""

    def __init__(self, args, n_actions=4, hidden=1024,
                 frame_height=84, frame_width=84, agent_history_length=4):
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
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length

        self.input = tf.placeholder(shape=[None, self.frame_height,
                                           self.frame_width, self.agent_history_length],
                                    dtype=tf.float32)
        # Normalizing the input
        self.inputscaled = (self.input - 127.5)/127.5
        self.raw_output = self.build_layers(self.inputscaled)
        # Splitting into value and advantage stream
        self.valuestream, self.advantagestream = tf.split(self.raw_output,2,-1)
        self.valuestream = tf.layers.flatten(self.valuestream)
        self.advantagestream = tf.layers.flatten(self.advantagestream)
        self.advantage = tf.layers.dense(
            inputs=self.advantagestream, units=self.n_actions,
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name="advantage")
        self.value = tf.layers.dense(
            inputs=self.valuestream, units=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name='value')

        # Combining value and advantage into Q-values as described above
        self.q_values = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
        self.action_prob = tf.nn.softmax(self.q_values)
        self.best_action = tf.argmax(self.q_values, 1)

        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.target_n_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.target_expert_q = tf.placeholder(shape=[None], dtype=tf.float32)

        self.expert_action = tf.placeholder(shape=[None], dtype=tf.int32)
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        # # Action that was performed
        # # Q value of the action that was performed
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)),
                               axis=1)
        self.loss, self.loss_per_sample = self.dqfd_loss()
        # Parameter updates
        # self.loss_per_sample = tf.losses.huber_loss(labels=self.target_q, predictions=self.Q, reduction=tf.losses.Reduction.NONE)
        # self.loss = tf.reduce_mean(self.loss_per_sample)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        self.update = self.optimizer.minimize(self.loss)



    def build_layers(self, state, units_1=24, units_2=24):
        with tf.variable_scope('select_net') as scope:
            c_names = ['select_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_i = tf.random_uniform_initializer(-0.1, 0.1)
            b_i = tf.constant_initializer(0.1)
            reg = tf.contrib.layers.l2_regularizer(scale=0.2)  # Note: only parameters in select-net need L2
            a_d = self.n_actions
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [a_d, units_1], initializer=w_i, collections=c_names, regularizer=reg)
                b1 = tf.get_variable('b1', [1, units_1], initializer=b_i, collections=c_names, regularizer=reg)
                dense1 = tf.nn.relu(tf.matmul(state, w1) + b1)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [units_1, units_2], initializer=w_i, collections=c_names, regularizer=reg)
                b2 = tf.get_variable('b2', [1, units_2], initializer=b_i, collections=c_names, regularizer=reg)
                dense2 = tf.nn.relu(tf.matmul(dense1, w2) + b2)
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [units_2, a_d], initializer=w_i, collections=c_names, regularizer=reg)
                b3 = tf.get_variable('b3', [1, a_d], initializer=b_i, collections=c_names, regularizer=reg)
                dense3 = tf.matmul(dense2, w3) + b3
            return dense3

    def loss_l(self, ae, a):
        return 0.0 if ae == a else 0.8

    def loss_jeq(self):
        expert_act_one_hot = tf.one_hot(self.expert_action, self.n_actions, dtype=tf.float32)

        #JEQ = max_{a in A}(Q(s,a) + l(a_e, a)) - Q(s, a_e)
        q_plus_margin = self.q_values + expert_act_one_hot
        q_plus_margin *= self.args.dqfd_margin
        max_q_plus_margin = softargmax(q_plus_margin)
        jeq = tf.losses.huber_loss(max_q_plus_margin, self.target_expert_q, reduction=tf.losses.Reduction.NONE)
        return jeq

    def dqfd_loss(self):
        l_dq = tf.losses.huber_loss(self.Q, self.target_q, reduction=tf.losses.Reduction.NONE)
        l_n_dq = tf.losses.huber_loss(self.Q, self.target_n_q, reduction=tf.losses.Reduction.NONE)
        l_jeq = self.loss_jeq()
        l_l2 = tf.reduce_sum([tf.reduce_mean(reg_l) for reg_l in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)])
        loss_per_sample = l_dq + l_n_dq + l_jeq
        loss = (tf.reduce_mean(loss_per_sample) + l_l2) * self.args.LAMBDA
        return loss, loss_per_sample



def learn(session, states, actions, rewards, new_states, terminal_flags, main_dqn, target_dqn, batch_size, gamma, args, expert=False):
    generated_states = (states - 127.5)/127.5
    generated_new_states = (new_states - 127.5)/127.5


    y_batch = np.zeros((batch_size, actions.shape[1]))



def learn(session, generated_states, generated_actions, generated_rewards, generated_new_states, generated_terminal_flags, main_dqn, target_dqn, batch_size, gamma, args):
    """
    Args:states
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
    # Draw a minibatch from the replay memory
    #weight = 1 - np.exp(policy_weight)/(np.exp(expert_weight) + np.exp(policy_weight))
    generated_states = (generated_states - 127.5)/127.5
    generated_new_states = (generated_new_states - 127.5)/127.5

    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:generated_new_states})
    # The target network estimates the Q-values (in the next state s', new_states is passed!)
    # for every transition in the minibatch
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:generated_new_states})
    double_q = q_vals[range(batch_size), arg_q_max]

    target_q = generated_rewards + (gamma*double_q * (1-generated_terminal_flags))
    # Gradient descend step to update the parameters of the main network
    loss_total = 0
    for i in range(args.td_iterations):
        loss, est_q, _ = session.run([main_dqn.loss_per_sample, main_dqn.Q, main_dqn.update],
                            feed_dict={main_dqn.input:generated_states,
                                        main_dqn.target_q:target_q,
                                        main_dqn.action:generated_actions})
        # for j in range(batch_size):
        #     print(j, loss[j], est_q[j], target_q[j])
        # quit()
        loss_total += loss
    loss_total/args.td_iterations
    return loss_total


def train(name="dqfd", priority=False):
    args = utils.argsparser()
    tf.random.set_random_seed(args.seed)
    np.random.seed(args.seed)


    MAX_EPISODE_LENGTH = args.max_eps_len       # Equivalent of 5 minutes of gameplay at 60 frames per second
    EVAL_FREQUENCY = args.eval_freq          # Number of frames the agent sees between evaluations
    EVAL_STEPS = args.eval_len               # Number of frames for one evaluation

    REPLAY_MEMORY_START_SIZE = args.replay_start_size # Number of completely random actions,
                                    # before the agent starts learning
    MAX_FRAMES = args.max_frames            # Total number of frames the agent sees
    MEMORY_SIZE = args.replay_mem_size            # Number of transitions stored in the replay memory
    NO_OP_STEPS = args.no_op_steps                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an
                                    # evaluation episode
    HIDDEN = args.hidden                    # Number of filters in the final convolutional layer. The output
                                    # has the shape (1,1,1024) which is split into two streams. Both
                                    # the advantage stream and value stream have the shape
                                    # (1,1,512). This is slightly different from the original
                                    # implementation but tests I did with the environment Pong
                                    # have shown that this way the score increases more quickly
                                    # Hessel et al. 2017 used 0.0000625
    atari = utils.Atari(args.env_id, args.stochastic_environment, NO_OP_STEPS)
    atari.env.seed(args.seed)
    # main DQN and target DQN networks:
    with tf.variable_scope('mainDQN'):
        MAIN_DQN = DQN(args, atari.env.action_space.n, HIDDEN)  # (★★)
    with tf.variable_scope('targetDQN'):
        TARGET_DQN = DQN(args, atari.env.action_space.n, HIDDEN)  # (★★)

    init = tf.global_variables_initializer()
    MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
    TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if priority:
        my_replay_memory = PriorityBuffer.PrioritizedReplayBuffer(MEMORY_SIZE, args.alpha)  # (★)
    else:
        my_replay_memory = PriorityBuffer.ReplayBuffer(MEMORY_SIZE)  # (★)

    network_updater = utils.TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
    action_getter = utils.ActionGetter(atari.env.action_space.n,
                                 replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                 max_frames=MAX_FRAMES,
                                 eps_initial=args.initial_exploration)
    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session(config=config)
    sess.run(init)

    frame_number = REPLAY_MEMORY_START_SIZE
    eps_number = 0

    if not os.path.exists("../" + args.gif_dir + "/" + name + "/" + args.env_id + "/"):
        os.makedirs("../" + args.gif_dir + "/" + name + "/" + args.env_id + "/")
    if not os.path.exists("../" + args.checkpoint_dir + "/" + name + "/" + args.env_id + "/"):
        os.makedirs("../" + args.checkpoint_dir + "/" + name + "/" + args.env_id + "/")
    if not os.path.exists("../" + args.expert_dir + "/" + name + "/" + args.env_id + "/"):
        os.makedirs("../" + args.expert_dir + "/" + name + "/" + args.env_id + "/")

    #if os.path.exists("../" + args.expert_dir + "/" + name + "/" + args.env_id + "/" + "expert_data.pkl"):
        #my_replay_memory.load_expert_data("../" + args.expert_dir + "/" + name + "/" + args.env_id + "/" + "expert_data.pkl")

    utils.build_initial_replay_buffer(sess, atari, my_replay_memory, action_getter, MAX_EPISODE_LENGTH, REPLAY_MEMORY_START_SIZE, MAIN_DQN, args)

    #Pretrain step ...
    #utils.train_step(sess, args, MAIN_DQN, TARGET_DQN, network_updater, action_getter, my_replay_memory, atari, 0, args.pretrain_bc_iter, learn, pretrain=True, priority=False)
    #utils.evaluate_model(sess, args, EVAL_STEPS, MAIN_DQN, action_getter, MAX_EPISODE_LENGTH, atari, frame_number, model_name=name, gif=True)
    episode_reward_list = []
    episode_len_list = []
    episode_loss_list = []
    episode_time_list = []

    print_iter = 25
    last_eval = 0
    while frame_number < MAX_FRAMES:
        eps_rw, eps_len, eps_loss, eps_time = utils.train_step(sess, args, MAIN_DQN, TARGET_DQN, network_updater, action_getter, my_replay_memory, atari, frame_number, learn, priority=priority)
        episode_reward_list.append(eps_rw)
        episode_len_list.append(eps_len)
        episode_loss_list.append(eps_loss)
        episode_time_list.append(eps_time)

        frame_number += eps_len
        eps_number += 1
        last_eval += eps_len
        if len(episode_len_list) % print_iter == 0:
            print("Last " + str(print_iter) + " Episodes Reward: ", np.mean(episode_reward_list[-print_iter:]))
            print("Last " + str(print_iter) + " Episodes Length: ", np.mean(episode_len_list[-print_iter:]))
            print("Last " + str(print_iter) + " Episodes Loss: ", np.mean(episode_loss_list[-print_iter:]))
            print("Last " + str(print_iter) + " Episodes Time: ", np.mean(episode_time_list[-print_iter:]))
            print("Last " + str(print_iter) + " Reward/Frames: ", np.sum(episode_reward_list[-print_iter:])/np.sum(episode_len_list[-print_iter:]))
            print("Total " + str(print_iter) + " Episode time: ", np.sum(episode_time_list[-print_iter:]))
            print("Current Exploration: ", action_getter.get_eps(frame_number))
            print("Frame Number: ", frame_number)
            print("Episode Number: ", eps_number)


        if EVAL_FREQUENCY < last_eval:
            last_eval = 0
            utils.evaluate_model(sess, args, EVAL_STEPS, MAIN_DQN, action_getter, MAX_EPISODE_LENGTH, atari,
                                 frame_number, model_name=name, gif=True)
            saver.save(sess, "../" + args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + "model",
                    global_step=frame_number)
train()
