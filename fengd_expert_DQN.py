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
import logger
import time
import pickle
import utils


class DQN:
    """Implements a Deep Q Network"""

    def __init__(self, n_actions=4, hidden=1024, learning_rate=0.00001, gamma =0.99,
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
        self.n_actions = n_actions
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length

        self.input = tf.placeholder(shape=[None, self.frame_height,
                                           self.frame_width, self.agent_history_length],
                                    dtype=tf.float32)
        # Normalizing the input
        self.inputscaled = self.input / 255

        # Convolutional layers
        self.conv1 = tf.layers.conv2d(
            inputs=self.inputscaled, filters=32, kernel_size=[8, 8], strides=4,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1')
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=2,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2')
        self.conv3 = tf.layers.conv2d(
            inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv3')
        self.conv4 = tf.layers.conv2d(
            inputs=self.conv3, filters=hidden, kernel_size=[7, 7], strides=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4')
        self.d = tf.layers.flatten(self.conv4)
        self.dense = tf.layers.dense(inputs = self.d,units = hidden,activation=tf.tanh,
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2), name="fc5" )
        self.q_values = tf.layers.dense(
            inputs=self.dense, units=n_actions,activation=tf.identity,
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name='value')

        # Combining value and advantage into Q-values as described above
        self.action_prob = tf.nn.softmax(self.q_values)
        self.best_action = tf.argmax(self.action_prob, 1)

        # The next lines perform the parameter update. This will be explained in detail later.

        # targetQ according to Bellman equation:
        # Q = r + gamma*max Q', calculated in the function learn()
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        # Action that was performed
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        self.expert_action = tf.placeholder(shape=[None], dtype =tf.int32)
        # Q value of the action that was performed
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)),
                               axis=1)

        # Parameter updates
        self.loss = tf.reduce_mean(math.gamma(1+gamma)*tf.math.exp(tf.losses.huber_loss(self.Q,self.target_q)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

        self.prob = tf.reduce_sum(tf.multiply(self.action_prob,
                                              tf.one_hot(self.expert_action, self.n_actions, dtype=tf.float32)),
                                         axis=1)
        self.best_Q = tf.reduce_sum(tf.multiply(self.q_values,
                                                tf.one_hot(self.best_action, self.n_actions, dtype=tf.float32)),
                               axis=1)
        self.expert_loss = tf.reduce_mean(-tf.log(self.prob+0.00001))# + a l2 reg to prevent overtraining too much
        self.expert_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate*0.25)
        self.expert_update =self.expert_optimizer.minimize(self.expert_loss)

def learn(session, dataset, replay_memory, main_dqn, target_dqn, batch_size, gamma):
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
    # Draw a minibatch from the replay memory
    states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch()
    obs,acs, _, _, _ = dataset.get_minibatch()
    obs = obs[:states.shape[0]]
    acs = acs[:states.shape[0]]
    # The main network estimates which action is best (in the next
    # state s', new_states is passed!)
    # for every transition in the minibatch
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
    # The target network estimates the Q-values (in the next state s', new_states is passed!)
    # for every transition in the minibatch
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})
    expert_arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input: obs})
    expert_q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:obs})
    double_q = q_vals[range(batch_size), arg_q_max]
    expert_q = expert_q_vals[range(batch_size),expert_arg_q_max]
    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = rewards + (gamma*double_q * (1-terminal_flags))
    # Gradient descend step to update the parameters of the main network
    loss, _ = session.run([main_dqn.loss, main_dqn.update],
                          feed_dict={main_dqn.input:states,
                                     main_dqn.target_q:target_q,
                                     main_dqn.action:actions})
    expert_loss, _ = session.run([main_dqn.expert_loss,main_dqn.expert_update],
                                 feed_dict={main_dqn.input:obs,
                                            main_dqn.expert_action:acs,
                                            main_dqn.target_q:expert_q})
    return loss,expert_loss

import argparse


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of DQN")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_dir', type=str, default='expert_dist_dqn_data/')
    parser.add_argument('--expert_file', type=str, default='expert_data.pkl')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='models/expert_dist_dqn/')
    parser.add_argument('--checkpoint_index', type=int, help='index of model to load', default=-1)
    parser.add_argument('--log_dir', help='the directory to save log file', default='logs/expert_dist_dqn/')
    parser.add_argument('--gif_dir', help='the directory to save GIFs file', default='GIFs/expert_dist_dqn/')
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    parser.add_argument('--num_sampled', type=int, help='Num Generated Sequence', default=1)
    parser.add_argument('--max_eps_len', type=int, help='Max Episode Length', default=18000)
    parser.add_argument('--eval_freq', type=int, help='Evaluation Frequency', default=100000)
    parser.add_argument('--eval_len', type=int, help='Max Episode Length', default=10000)
    parser.add_argument('--target_update_freq', type=int, help='Max Episode Length', default=10000)
    parser.add_argument('--replay_start_size', type=int, help='Max Episode Length', default=50000)
    parser.add_argument('--max_frames', type=int, help='Max Episode Length', default=50000000)
    parser.add_argument('--replay_mem_size', type=int, help='Max Episode Length', default=1000000)
    parser.add_argument('--no_op_steps', type=int, help='Max Episode Length', default=10)
    parser.add_argument('--update_freq', type=int, help='Max Episode Length', default=4)
    parser.add_argument('--hidden', type=int, help='Max Episode Length', default=1024)
    parser.add_argument('--batch_size', type=int, help='Max Episode Length', default=32)
    parser.add_argument('--gamma', type=float, help='Max Episode Length', default=0.99)
    parser.add_argument('--lr', type=float, help='Max Episode Length', default=0.00001)
    parser.add_argument('--stochastic_exploration', type=str, default="False")
    parser.add_argument('--initial_exploration', type=float, help='Amount of exploration at start', default=1.0)
    parser.add_argument('--env_id', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--stochastic', type=str, choices=['True', 'False'], default='True')
    parser.add_argument('--load_model_dir', type=str, default='models/')
    parser.add_argument('--load_expert_data', type=str, default='data/')

    return parser.parse_args()

args = utils.argsparser()
tf.random.set_random_seed(args.seed)
np.random.seed(args.seed)

tf.reset_default_graph()
# Control parameters
if args.task == "train":
    utils.train(args, DQN, learn, "basic_fengdi_dist_expert_dqn", expert=True)
elif args.task == "evaluate":
    utils.sample(args, DQN, "basic_fengdi_dist_expert_dqn", save=False)
else:
    utils.sample(args, DQN, "basic_fengdi_dist_expert_dqn")