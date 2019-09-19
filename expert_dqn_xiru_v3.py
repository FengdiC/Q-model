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

    def __init__(self, n_actions=4, hidden=1024, learning_rate=0.00001, bc_learning_rate=0.001, max_ent_coef=1.0,gamma =0.99,
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
        self.bc_learning_rate = bc_learning_rate
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length

        self.decay = tf.placeholder(shape=[None], dtype=tf.float32)
        self.input = tf.placeholder(shape=[None, self.frame_height,
                                           self.frame_width, self.agent_history_length],
                                    dtype=tf.float32)
        self.generated_input = tf.placeholder(shape=[None, self.frame_height,
                                           self.frame_width, self.agent_history_length],
                                    dtype=tf.float32)
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.expert_weights = tf.placeholder(shape=[None], dtype=tf.float32)

        self.q_values, self.action_preference = self.build_graph(self.input, hidden, n_actions)
        self.action_prob_q = tf.nn.softmax(self.q_values)
        self.action_prob_expert = tf.nn.softmax(self.action_preference)
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)),
                               axis=1)
        t_vars = tf.trainable_variables()
        q_value_vars = []
        expert_vars = []
        for v in t_vars:
            if 'ExpertPref' in v.name:
                expert_vars.append(v)
            else:
                q_value_vars.append(v)

        # Parameter updates
        self.loss = tf.reduce_mean(math.gamma(1+gamma)*tf.math.exp(tf.losses.huber_loss(self.Q,self.target_q)) - math.gamma(1+gamma))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss, var_list=q_value_vars)


        #The loss based on expert trajectories ...
        self.expert_action = tf.placeholder(shape=[None], dtype=tf.int32)
        self.expert_prob = tf.reduce_sum(tf.multiply(self.action_prob_expert,
                                              tf.one_hot(self.expert_action, self.n_actions, dtype=tf.float32)),
                                         axis=1)
        #self.behavior_cloning_loss = tf.reduce_mean(-tf.log(self.expert_prob  +0.00001))# + a l2 reg to prevent overtraining too much
        self.behavior_cloning_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.action_preference, labels=tf.one_hot(self.expert_action, self.n_actions, dtype=tf.float32)))
        self.bc_optimizer = tf.train.AdamOptimizer(learning_rate=self.bc_learning_rate)
        #self.regularization = tf.reduce_mean(tf.reduce_sum(self.action_prob_expert * tf.log(self.action_prob_expert),axis=1))
        #self.behavior_cloning_loss += self.regularization
        self.bc_optimizer = tf.train.AdamOptimizer(learning_rate=self.bc_learning_rate)
        self.bc_update =self.bc_optimizer.minimize(self.behavior_cloning_loss, var_list=expert_vars)

        self.expert_loss = tf.reduce_mean(tf.reduce_sum(-tf.log(self.action_prob_q + 0.00001) * self.action_prob_expert, axis=1) * self.decay)
        self.expert_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.expert_update =self.expert_optimizer.minimize(self.expert_loss, var_list=q_value_vars)

        self.action_prob = self.action_prob_q
        self.best_action = tf.argmax(self.action_prob, 1)

    def build_graph(self, input, hidden, n_actions, reuse=False):
        # Convolutional layers
        self.scaled_input = input / 255
        with tf.variable_scope("Qvalue"):
            self.conv1 = tf.layers.conv2d(
                inputs=self.scaled_input, filters=32, kernel_size=[8, 8], strides=4,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1', reuse=reuse)
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=2,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2', reuse=reuse)
            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=1,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv3', reuse=reuse)
            # self.conv4 = tf.layers.conv2d(
            #     inputs=self.conv3, filters=hidden, kernel_size=[7, 7], strides=1,
            #     kernel_initializer=tf.variance_scaling_initializer(scale=2),
            #     padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4', reuse=reuse)
            self.d = tf.layers.flatten(self.conv3)
            self.dense = tf.layers.dense(inputs = self.d,units = hidden,activation=tf.tanh,
                                         kernel_initializer=tf.variance_scaling_initializer(scale=2), name="fc5", reuse=reuse)
            q_values = tf.layers.dense(
                inputs=self.dense, units=n_actions,activation=tf.identity,
                kernel_initializer=tf.variance_scaling_initializer(scale=2), name='value', reuse=reuse)

        with tf.variable_scope("ExpertPref"):
            self.expert_conv1 = tf.layers.conv2d(
                inputs=self.scaled_input, filters=32, kernel_size=[8, 8], strides=4,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1', reuse=reuse)
            self.expert_conv2 = tf.layers.conv2d(
                inputs=self.expert_conv1, filters=64, kernel_size=[4, 4], strides=2,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2', reuse=reuse)
            self.expert_conv3 = tf.layers.conv2d(
                inputs=self.expert_conv2, filters=64, kernel_size=[3, 3], strides=1,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv3', reuse=reuse)
            # self.expert_conv4 = tf.layers.conv2d(
            #     inputs=self.expert_conv3, filters=hidden, kernel_size=[7, 7], strides=1,
            #     kernel_initializer=tf.variance_scaling_initializer(scale=2),
            #     padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4', reuse=reuse)
            self.expert_d = tf.layers.flatten(self.expert_conv3)
            self.expert_dense = tf.layers.dense(inputs =self.expert_d,units = hidden,activation=tf.tanh,
                                         kernel_initializer=tf.variance_scaling_initializer(scale=2), name="fc5", reuse=reuse)
            action_preference = tf.layers.dense(
                inputs=self.expert_dense, units=n_actions,activation=tf.identity,
                kernel_initializer=tf.variance_scaling_initializer(scale=2), name='action_preference', reuse=reuse)
        return q_values, action_preference

def train_bc(session, dataset, replay_dataset, main_dqn, pretrain=False):
    states, actions, _, _, _, weights = utils.get_minibatch(dataset)
    gen_states_1, _, _, _, _ = replay_dataset.get_minibatch() #Generated trajectories
    gen_states_2, _, _, _, _ = replay_dataset.get_minibatch() #Generated trajectories
    gen_states = np.concatenate([gen_states_1, gen_states_2], axis=0)
    expert_loss, _ = session.run([main_dqn.behavior_cloning_loss, main_dqn.bc_update],
                              feed_dict={main_dqn.input: states,
                                         main_dqn.expert_action: actions,
                                         main_dqn.expert_weights: weights,
                                         main_dqn.generated_input: gen_states})

    return expert_loss

def learn(session, dataset, replay_memory, main_dqn, target_dqn, batch_size, gamma, args):
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
    expert_states, expert_actions, expert_rewards, expert_new_states, expert_terminal_flags, weights = utils.get_minibatch(dataset)  #Expert trajectories
    generated_states, generated_actions, generated_rewards, generated_new_states, generated_terminal_flags = replay_memory.get_minibatch() #Generated trajectories

    # The main network estimates which action is best (in the next
    # state s', new_states is passed!)
    # for every transition in the minibatch

    # next_states = np.concatenate((expert_new_states, generated_new_states), axis=0)
    # combined_terminal_flags = np.concatenate((expert_terminal_flags, generated_terminal_flags), axis=0)
    # combined_rewards = np.concatenate((expert_rewards, generated_rewards), axis=0)
    # combined_actions = np.concatenate((expert_actions, generated_actions), axis=0)
    # combined_states = np.concatenate((expert_states, generated_states), axis=0)

    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:generated_new_states})
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:generated_new_states})
    double_q = q_vals[range(batch_size), arg_q_max]

    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = generated_rewards + (gamma*double_q * (1-generated_terminal_flags))

    # Gradient descend step to update the parameters of the main network
    for i in range(1):
        loss, _ = session.run([main_dqn.loss, main_dqn.update],
                            feed_dict={main_dqn.input:generated_states,
                                        main_dqn.target_q:target_q,
                                        main_dqn.action:generated_actions})

    expert_loss, _ = session.run([main_dqn.expert_loss,main_dqn.expert_update],
                                 feed_dict={main_dqn.input:expert_states,
                                            main_dqn.generated_input:generated_states,
                                            main_dqn.expert_action:expert_actions,
                                            main_dqn.expert_weights:weights,
                                            main_dqn.decay:np.exp(-replay_memory.total_count/args.decay_rate) + np.zeros((expert_states.shape[0],))})
    return loss,expert_loss

args = utils.argsparser()
tf.random.set_random_seed(args.seed)
np.random.seed(args.seed)

tf.reset_default_graph()
# Control parameters
if args.task == "train":
    utils.train(args, DQN, learn, "expert_dist_dqn", expert=True, bc_training=train_bc, pretrain_iters=args.pretrain_bc_iter)
elif args.task == "evaluate":
    utils.sample(args, DQN, "expert_dist_dqn", save=False)
elif args.task == "log":
    utils.generate_figures("expert_dist_dqn")
else:
    utils.sample(args, DQN, "expert_dist_dqn")