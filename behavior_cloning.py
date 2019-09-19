dist_DQN.py                                                                                         0000644 0010623 0003410 00000016023 13540540367 012477  0                                                                                                    ustar   xiruzhu                         meger                                                                                                                                                                                                                  import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')
import gym
from tqdm import tqdm
import imageio
from skimage.transform import resize
import logger
import math
import time
import utils
import pickle
class DQN:
    """Implements a Deep Q Network"""
    def __init__(self, n_actions=4, hidden=1024, learning_rate=0.00001, gamma=0.99,
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
        # self.conv4 = tf.layers.conv2d(
        #     inputs=self.conv3, filters=hidden, kernel_size=[7, 7], strides=1,
        #     kernel_initializer=tf.variance_scaling_initializer(scale=2),
        #     padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4')
        self.d = tf.layers.flatten(self.conv3)
        self.dense = tf.layers.dense(inputs=self.d, units=hidden, activation=tf.tanh,
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2), name="fc5")
        self.q_values = tf.layers.dense(
            inputs=self.dense, units=n_actions, activation=tf.identity,
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
        self.expert_action = tf.placeholder(shape=[None], dtype=tf.int32)
        # Q value of the action that was performed
        self.Q = tf.reduce_sum(
            tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)),
            axis=1)

        # Parameter updates
        self.loss = tf.reduce_mean(math.gamma(1 + gamma) * tf.math.exp(tf.losses.huber_loss(self.Q, self.target_q)) - math.gamma(1+gamma))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

        self.prob = tf.reduce_sum(tf.multiply(self.action_prob,
                                              tf.one_hot(self.expert_action, self.n_actions, dtype=tf.float32)),
                                  axis=1)
        self.best_Q = tf.reduce_sum(tf.multiply(self.q_values,
                                                tf.one_hot(self.best_action, self.n_actions, dtype=tf.float32)),
                                    axis=1)
        self.expert_loss = -tf.log(self.prob + 0.001)
        self.expert_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate * 0.25)
        self.expert_update = self.expert_optimizer.minimize(self.expert_loss)


def learn(session, replay_memory, main_dqn, target_dqn, batch_size, gamma):
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
    # The main network estimates which action is best (in the next
    # state s', new_states is passed!)
    # for every transition in the minibatch
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
    # The target network estimates the Q-values (in the next state s', new_states is passed!)
    # for every transition in the minibatch
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})
    double_q = q_vals[range(batch_size), arg_q_max]
    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = rewards + (gamma*double_q * (1-terminal_flags))
    # Gradient descend step to update the parameters of the main network
    loss, _ = session.run([main_dqn.loss, main_dqn.update],
                          feed_dict={main_dqn.input:states,
                                     main_dqn.target_q:target_q,
                                     main_dqn.action:actions})
    return loss

args = utils.argsparser()
tf.random.set_random_seed(args.seed)
np.random.seed(args.seed)

tf.reset_default_graph()
# Control parameters
if args.task == "train":
    utils.train(args, DQN, learn, "dist_dqn")
elif args.task == "evaluate":
    utils.sample(args, DQN, "dist_dqn", save=False)
elif args.task == "log":
    utils.generate_figures("dist_dqn")
else:
    utils.sample(args, DQN, "dist_dqn")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             DQN.py                                                                                              0000644 0010623 0003410 00000015361 13540606234 011454  0                                                                                                    ustar   xiruzhu                         meger                                                                                                                                                                                                                  import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')
import gym
from tqdm import tqdm
import imageio
from skimage.transform import resize
import logger
import time
import pickle
import utils

class DQN:
    """Implements a Deep Q Network"""

    def __init__(self, n_actions=4, hidden=1024, learning_rate=0.00001,
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
        # self.conv4 = tf.layers.conv2d(
        #     inputs=self.conv3, filters=hidden, kernel_size=[7, 7], strides=1,
        #     kernel_initializer=tf.variance_scaling_initializer(scale=2),
        #     padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4')
        self.d = tf.layers.flatten(self.conv3)
        self.dense = tf.layers.dense(inputs = self.d,units = hidden,
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2), name="fc5" )

        # Splitting into value and advantage stream
        self.valuestream, self.advantagestream = tf.split(self.dense,2,-1)
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

        # The next lines perform the parameter update. This will be explained in detail later.

        # targetQ according to Bellman equation:
        # Q = r + gamma*max Q', calculated in the function learn()
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        # Action that was performed
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        # Q value of the action that was performed
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)),
                               axis=1)
        # Parameter updates
        self.loss = tf.reduce_mean(tf.losses.huber_loss(labels=self.target_q, predictions=self.Q))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

def learn(session, replay_memory, main_dqn, target_dqn, batch_size, gamma):
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
    # The main network estimates which action is best (in the next
    # state s', new_states is passed!)
    # for every transition in the minibatch
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
    # The target network estimates the Q-values (in the next state s', new_states is passed!)
    # for every transition in the minibatch
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})
    double_q = q_vals[range(batch_size), arg_q_max]
    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = rewards + (gamma*double_q * (1-terminal_flags))
    # Gradient descend step to update the parameters of the main network
    loss, _ = session.run([main_dqn.loss, main_dqn.update],
                          feed_dict={main_dqn.input:states,
                                     main_dqn.target_q:target_q,
                                     main_dqn.action:actions})
    return loss

args = utils.argsparser()
tf.random.set_random_seed(args.seed)
np.random.seed(args.seed)

tf.reset_default_graph()
# Control parameters
if args.task == "train":
    utils.train(args, DQN, learn, "dqn")
elif args.task == "evaluate":
    utils.sample(args, DQN, "dqn", save=False)
elif args.task == "log":
    utils.generate_figures("dqn")
else:
    utils.sample(args, DQN, "dqn")                                                                                                                                                                                                                                                                               expert_DQN.py                                                                                       0000644 0010623 0003410 00000030425 13540540614 013040  0                                                                                                    ustar   xiruzhu                         meger                                                                                                                                                                                                                  import tensorflow as tf
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
        self.bc_optimizer = tf.train.AdamOptimizer(learning_rate=10 * self.learning_rate)
        #self.regularization = tf.reduce_mean(tf.reduce_sum(self.action_prob_expert * tf.log(self.action_prob_expert),axis=1))
        #self.behavior_cloning_loss += self.regularization
        self.bc_optimizer = tf.train.AdamOptimizer(learning_rate=self.bc_learning_rate)
        self.bc_update =self.bc_optimizer.minimize(self.behavior_cloning_loss, var_list=expert_vars)

        self.expert_loss = tf.reduce_mean(tf.reduce_sum(-tf.log(self.action_prob_q + 0.00001) * self.action_prob_expert, axis=1))
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

def learn(session, dataset, replay_memory, main_dqn, target_dqn, batch_size, gamma):
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
                                            main_dqn.expert_weights:weights})
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
    utils.sample(args, DQN, "expert_dist_dqn")                                                                                                                                                                                                                                           expert_dqn_xiru_v1.py                                                                               0000644 0010623 0003410 00000030620 13540540305 014647  0                                                                                                    ustar   xiruzhu                         meger                                                                                                                                                                                                                  import tensorflow as tf
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

    def __init__(self, n_actions=4, hidden=1024, learning_rate=0.00001, bc_learning_rate=0.001, max_ent_coef=1.0, gamma =0.99,
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
        self.regularization = tf.reduce_mean(tf.reduce_sum(self.action_prob_expert * tf.log(self.action_prob_expert + 0.000001), axis=1))
        self.behavior_cloning_loss += max_ent_coef * self.regularization

        #self.regularization = tf.reduce_mean(tf.reduce_sum(self.action_prob_expert * tf.log(self.action_prob_expert),axis=1))
        #self.behavior_cloning_loss += self.regularization
        self.bc_optimizer = tf.train.AdamOptimizer(learning_rate=self.bc_learning_rate)
        self.bc_update =self.bc_optimizer.minimize(self.behavior_cloning_loss, var_list=expert_vars)

        self.expert_loss = tf.reduce_mean(tf.reduce_sum(-tf.log(self.action_prob_q + 0.00001) * self.action_prob_expert, axis=1))
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

def learn(session, dataset, replay_memory, main_dqn, target_dqn, batch_size, gamma):
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
                                            main_dqn.expert_weights:weights})
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
    utils.sample(args, DQN, "expert_dist_dqn")                                                                                                                expert_dqn_xiru_v2.py                                                                               0000644 0010623 0003410 00000030661 13540540316 014657  0                                                                                                    ustar   xiruzhu                         meger                                                                                                                                                                                                                  import tensorflow as tf
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

        _, self.generated_action_preference = self.build_graph(self.generated_input, hidden, n_actions, reuse=True)
        self.generated_action_prob = tf.nn.softmax(self.generated_action_preference)
        self.regularization = tf.reduce_mean(tf.reduce_sum(self.generated_action_prob * tf.log(self.generated_action_prob),axis=1))
        self.behavior_cloning_loss += max_ent_coef * self.regularization
        self.bc_optimizer = tf.train.AdamOptimizer(learning_rate=self.bc_learning_rate)
        self.bc_update =self.bc_optimizer.minimize(self.behavior_cloning_loss, var_list=expert_vars)

        self.expert_loss = tf.reduce_mean(tf.reduce_sum(-tf.log(self.action_prob_q + 0.00001) * self.action_prob_expert, axis=1))
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
    expert_loss, reg, _ = session.run([main_dqn.behavior_cloning_loss, main_dqn.action_prob, main_dqn.bc_update],
                              feed_dict={main_dqn.input: states,
                                         main_dqn.expert_action: actions,
                                         main_dqn.expert_weights: weights,
                                         main_dqn.generated_input: gen_states})
    return expert_loss

def learn(session, dataset, replay_memory, main_dqn, target_dqn, batch_size, gamma):
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
                                            main_dqn.expert_weights:weights})
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
    utils.sample(args, DQN, "expert_dist_dqn")                                                                               fengd_expert_DQN.py                                                                                 0000644 0010623 0003410 00000017655 13540441252 014213  0                                                                                                    ustar   xiruzhu                         meger                                                                                                                                                                                                                  import tensorflow as tf
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
        #self.expert_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.q_values, labels=tf.one_hot(self.expert_action, self.n_actions, dtype=tf.float32)))
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

args = utils.argsparser()
tf.random.set_random_seed(args.seed)
np.random.seed(args.seed)

tf.reset_default_graph()
# Control parameters
if args.task == "train":
    utils.train(args, DQN, learn, "basic_fengdi_dist_expert_dqn", expert=True)
elif args.task == "evaluate":
    utils.sample(args, DQN, "basic_fengdi_dist_expert_dqn", save=False)
elif args.task == "log":
    utils.generate_figures("basic_fengdi_dist_expert_dqn")
else:
    utils.sample(args, DQN, "basic_fengdi_dist_expert_dqn")                                                                                   logger.py                                                                                           0000644 0010623 0003410 00000034721 13540441252 012307  0                                                                                                    ustar   xiruzhu                         meger                                                                                                                                                                                                                  import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
from collections import defaultdict
from contextlib import contextmanager

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50

class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError

class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError

class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, 'wt')
            self.own_file = True
        else:
            assert hasattr(filename_or_file, 'read'), 'expected file or str, got %s'%filename_or_file
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if hasattr(val, '__float__'):
                valstr = '%-8.3g' % val
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print('WARNING: tried to write empty key-value dict')
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        maxlen = 30
        return s[:maxlen-3] + '...' if len(s) > maxlen else s

    def writeseq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1: # add space unless this is the last one
                self.file.write(' ')
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()

class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, 'wt')

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, 'dtype'):
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()

class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, 'w+t')
        self.keys = []
        self.sep = ','

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(k)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write('\n')
        self.file.flush()

    def close(self):
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """
    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = 'events'
        path = osp.join(osp.abspath(dir), prefix)
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat
        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        def summary_val(k, v):
            kwargs = {'tag': k, 'simple_value': float(v)}
            return self.tf.Summary.Value(**kwargs)
        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = self.step # is there any reason why you'd want to specify the step?
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None

def make_output_format(format, ev_dir, log_suffix=''):
    os.makedirs(ev_dir, exist_ok=True)
    if format == 'stdout':
        return HumanOutputFormat(sys.stdout)
    elif format == 'log':
        return HumanOutputFormat(osp.join(ev_dir, 'log%s.txt' % log_suffix))
    elif format == 'json':
        return JSONOutputFormat(osp.join(ev_dir, 'progress%s.json' % log_suffix))
    elif format == 'csv':
        return CSVOutputFormat(osp.join(ev_dir, 'progress%s.csv' % log_suffix))
    elif format == 'tensorboard':
        return TensorBoardOutputFormat(osp.join(ev_dir, 'tb%s' % log_suffix))
    else:
        raise ValueError('Unknown format specified: %s' % (format,))

# ================================================================
# API
# ================================================================

def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    get_current().logkv(key, val)

def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    get_current().logkv_mean(key, val)

def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)

def dumpkvs():
    """
    Write all of the diagnostics from the current iteration
    """
    return get_current().dumpkvs()

def getkvs():
    return get_current().name2val


def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    get_current().log(*args, level=level)

def debug(*args):
    log(*args, level=DEBUG)

def info(*args):
    log(*args, level=INFO)

def warn(*args):
    log(*args, level=WARN)

def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    get_current().set_level(level)

def set_comm(comm):
    get_current().set_comm(comm)

def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return get_current().get_dir()

record_tabular = logkv
dump_tabular = dumpkvs

@contextmanager
def profile_kv(scopename):
    logkey = 'wait_' + scopename
    tstart = time.time()
    try:
        yield
    finally:
        get_current().name2val[logkey] += time.time() - tstart

def profile(n):
    """
    Usage:
    @profile("my_func")
    def my_func(): code
    """
    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with profile_kv(n):
                return func(*args, **kwargs)
        return func_wrapper
    return decorator_with_name


# ================================================================
# Backend
# ================================================================

def get_current():
    if Logger.CURRENT is None:
        _configure_default_logger()

    return Logger.CURRENT


class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
                    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, comm=None):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval*cnt/(cnt+1) + val/(cnt+1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        if self.comm is None:
            d = self.name2val
        else:
            from baselines.common import mpi_util
            d = mpi_util.mpi_weighted_mean(self.comm,
                {name : (val, self.name2cnt.get(name, 1))
                    for (name, val) in self.name2val.items()})
            if self.comm.rank != 0:
                d['dummy'] = 1 # so we don't get a warning about empty dict
        out = d.copy() # Return the dict for unit testing purposes
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(d)
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def set_comm(self, comm):
        self.comm = comm

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))

def get_rank_without_mpi_import():
    # check environment variables here instead of importing mpi4py
    # to avoid calling MPI_Init() when this module is imported
    for varname in ['PMI_RANK', 'OMPI_COMM_WORLD_RANK']:
        if varname in os.environ:
            return int(os.environ[varname])
    return 0


def configure(dir=None, format_strs=None, comm=None, log_suffix=''):
    """
    If comm is provided, average all numerical stats across that comm
    """
    if dir is None:
        dir = os.getenv('OPENAI_LOGDIR')
    if dir is None:
        dir = osp.join(tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(dir, str)
    dir = os.path.expanduser(dir)
    os.makedirs(os.path.expanduser(dir), exist_ok=True)

    rank = get_rank_without_mpi_import()
    if rank > 0:
        log_suffix = log_suffix + "-rank%03i" % rank

    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv('OPENAI_LOG_FORMAT', 'stdout,log,csv').split(',')
        else:
            format_strs = os.getenv('OPENAI_LOG_FORMAT_MPI', 'log').split(',')
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
    if output_formats:
        log('Logging to %s'%dir)

def _configure_default_logger():
    configure()
    Logger.DEFAULT = Logger.CURRENT

def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log('Reset logger')

@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    prevlogger = Logger.CURRENT
    configure(dir=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        Logger.CURRENT.close()
        Logger.CURRENT = prevlogger

# ================================================================

def _demo():
    info("hi")
    debug("shouldn't appear")
    set_level(DEBUG)
    debug("should appear")
    dir = "/tmp/testlogging"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    configure(dir=dir)
    logkv("a", 3)
    logkv("b", 2.5)
    dumpkvs()
    logkv("b", -2.5)
    logkv("a", 5.5)
    dumpkvs()
    info("^^^ should see a = 5.5")
    logkv_mean("b", -22.5)
    logkv_mean("b", -44.4)
    logkv("a", 5.5)
    dumpkvs()
    info("^^^ should see b = -33.3")

    logkv("b", -2.5)
    dumpkvs()

    logkv("a", "longasslongasslongasslongasslongasslongassvalue")
    dumpkvs()


# ================================================================
# Readers
# ================================================================

def read_json(fname):
    import pandas
    ds = []
    with open(fname, 'rt') as fh:
        for line in fh:
            ds.append(json.loads(line))
    return pandas.DataFrame(ds)

def read_csv(fname):
    import pandas
    return pandas.read_csv(fname, index_col=None, comment='#')

def read_tb(path):
    """
    path : a tensorboard file OR a directory, where we will find all TB files
           of the form events.*
    """
    import pandas
    import numpy as np
    from glob import glob
    import tensorflow as tf
    if osp.isdir(path):
        fnames = glob(osp.join(path, "events.*"))
    elif osp.basename(path).startswith("events."):
        fnames = [path]
    else:
        raise NotImplementedError("Expected tensorboard file or directory containing them. Got %s"%path)
    tag2pairs = defaultdict(list)
    maxstep = 0
    for fname in fnames:
        for summary in tf.train.summary_iterator(fname):
            if summary.step > 0:
                for v in summary.summary.value:
                    pair = (summary.step, v.simple_value)
                    tag2pairs[v.tag].append(pair)
                maxstep = max(summary.step, maxstep)
    data = np.empty((maxstep, len(tag2pairs)))
    data[:] = np.nan
    tags = sorted(tag2pairs.keys())
    for (colidx,tag) in enumerate(tags):
        pairs = tag2pairs[tag]
        for (step, value) in pairs:
            data[step-1, colidx] = value
    return pandas.DataFrame(data, columns=tags)

if __name__ == "__main__":
    _demo()                                               play.py                                                                                             0000644 0010623 0003410 00000016416 13540605223 011776  0                                                                                                    ustar   xiruzhu                         meger                                                                                                                                                                                                                  import gym
import pygame
import matplotlib
import argparse
from gym import logger
import pickle
import utils

try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.warn('failed to set matplotlib backend, plotting will not work: %s' % str(e))
    plt = None

from collections import deque
from pygame.locals import VIDEORESIZE
import tensorflow as tf

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))


def play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
    """Allows one to play the game using keyboard.
    To simply play the game use:
        play(gym.make("Pong-v4"))
    Above code works also if env is wrapped, so it's particularly useful in
    verifying that the frame-level preprocessing does not render the game
    unplayable.
    If you wish to plot real time statistics as you play, you can use
    gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
    for last 5 second of gameplay.
        def callback(obs_t, obs_tp1, action, rew, done, info):
            return [rew,]
        plotter = PlayPlot(callback, 30 * 5, ["reward"])
        env = gym.make("Pong-v4")
        play(env, callback=plotter.callback)
    Arguments
    ---------
    env: gym.Env
        Environment to use for playing.
    transpose: bool
        If True the output of observation is transposed.
        Defaults to true.
    fps: int
        Maximum number of steps of the environment to execute every second.
        Defaults to 30.
    zoom: float
        Make screen edge this many times bigger
    callback: lambda or None
        Callback if a callback is provided it will be executed after
        every step. It takes the following input:
            obs_t: observation before performing action
            obs_tp1: observation after performing action
            action: action that was executed
            rew: reward that was received
            done: whether the environment is done or not
            info: debug info
    keys_to_action: dict: tuple(int) -> int or None
        Mapping from keys pressed to action performed.
        For example if pressed 'w' and space at the same time is supposed
        to trigger action number 2 then key_to_action dict would look like this:
            {
                # ...
                sorted(ord('w'), ord(' ')) -> 2
                # ...
            }
        If None, default key_to_action mapping for that env is used, if provided.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    data_list = []
    obs = env.reset(sess)
    rendered = env.env.render(mode='rgb_array')

    if keys_to_action is None:
        if hasattr(env.env, 'get_keys_to_action'):
            keys_to_action = env.env.get_keys_to_action()
        elif hasattr(env.env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.env.unwrapped.get_keys_to_action()
        else:
            assert False, env.env.spec.id + " does not have explicit key to action mapping, " + \
                          "please specify one manually"
    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

    video_size = [rendered.shape[1], rendered.shape[0]]
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = True

    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()
    count = 0
    num_traj = 0
    while running:
        if env_done and count > 0:
            env_done = False
            num_traj += 1
            obs = env.reset(sess)
            print(num_traj, count)
            replay_mem = utils.ReplayMemory(len(data_list))
            for i in range(len(data_list)):
                action = data_list[i][0]
                obs = data_list[i][1]
                rew = data_list[i][2]
                terminal = data_list[i][3]
                replay_mem.add_experience(action=action,
                                          frame=obs[:, :, 0],
                                          reward=rew,
                                          terminal=terminal)

            pickle.dump(replay_mem, open("human_trajectories.pkl", "wb"), protocol=4)

        else:
            action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
            obs, rew, env_done, terminal, frame = env.step(sess, action)
            data_list.append([action, obs, rew, terminal])

            count += 1
        if obs is not None:
            rendered = env.env.render(mode='rgb_array')
            display_arr(screen, rendered, transpose=transpose, video_size=video_size)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                print(video_size)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


class PlayPlot(object):
    def __init__(self, callback, horizon_timesteps, plot_names):
        self.data_callback = callback
        self.horizon_timesteps = horizon_timesteps
        self.plot_names = plot_names

        assert plt is not None, "matplotlib backend failed, plotting will not work"

        num_plots = len(self.plot_names)
        self.fig, self.ax = plt.subplots(num_plots)
        if num_plots == 1:
            self.ax = [self.ax]
        for axis, name in zip(self.ax, plot_names):
            axis.set_title(name)
        self.t = 0
        self.cur_plot = [None for _ in range(num_plots)]
        self.data = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

    def callback(self, obs_t, obs_tp1, action, rew, done, info):
        points = self.data_callback(obs_t, obs_tp1, action, rew, done, info)
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1

        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t

        for i, plot in enumerate(self.cur_plot):
            if plot is not None:
                plot.remove()
            self.cur_plot[i] = self.ax[i].scatter(range(xmin, xmax), list(self.data[i]), c='blue')
            self.ax[i].set_xlim(xmin, xmax)
        plt.pause(0.000001)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutDeterministic-v4', help='Define Environment')
    args = parser.parse_args()
    #env = gym.make(args.env)
    env = utils.Atari(args.env, False)
    play(env, zoom=2, fps=60)


if __name__ == '__main__':
    main()
                                                                                                                                                                                                                                                  test_expert.py                                                                                      0000644 0010623 0003410 00000002357 13540540614 013400  0                                                                                                    ustar   xiruzhu                         meger                                                                                                                                                                                                                  import numpy as np
import tensorflow as tf
import pickle
from skimage.transform import resize
import imageio

dataset = pickle.load(open('/network/home/chefengd/Documents/Q-model/data/expert_dist_dqn/SeaquestDeterministic-v4/expert_data.pkl_30','rb'))


def generate_gif(frames_for_gif):
    """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """
    #for idx, frame_idx in enumerate(frames_for_gif):
        #frames_for_gif[idx] = resize(frame_idx, (420, 320, 3),
        #                             preserve_range=True, order=0).astype(np.uint8)

    imageio.mimsave('test_expert.gif',
                    frames_for_gif, duration=1 / 5)

print("number of frames in total: ", dataset.count)

num_eps = 0
accumulated_reward = 0
for i in range(dataset.count):
    accumulated_reward += dataset.rewards[i]
    if dataset.terminal_flags[i]:
        print("{0} episode trajectory reward: ".format(num_eps),accumulated_reward)
        accumulated_reward = 0

generate_gif(dataset.frames[:20000])                                                                                                                                                                                                                                                                                 utils.py                                                                                            0000644 0010623 0003410 00000133266 13540574066 012206  0                                                                                                    ustar   xiruzhu                         meger                                                                                                                                                                                                                  import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')
import gym
from tqdm import tqdm
import imageio
from skimage.transform import resize
import logger
import math
import time
import pickle
from numpy import genfromtxt
import csv

import argparse
def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of DQN")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_dir', type=str, default='data/')
    parser.add_argument('--expert_file', type=str, default='expert_data.pkl')
    parser.add_argument('--expert_file_path', type=str, default='None')

    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='models/')
    parser.add_argument('--checkpoint_index', type=int, help='index of model to load', default=-1)
    parser.add_argument('--checkpoint_file_path', type=str, default='None')
    parser.add_argument('--special_tag', type=str, default='')

    parser.add_argument('--log_dir', help='the directory to save log file', default='logs/')
    parser.add_argument('--gif_dir', help='the directory to save GIFs file', default='GIFs/')
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
    parser.add_argument('--update_freq', type=int, help='Max Episode Length', default=6)
    parser.add_argument('--hidden', type=int, help='Max Episode Length', default=512)
    parser.add_argument('--batch_size', type=int, help='Max Episode Length', default=32)
    parser.add_argument('--gamma', type=float, help='Max Episode Length', default=0.99)
    parser.add_argument('--lr', type=float, help='Max Episode Length', default=0.0000625)
    parser.add_argument('--lr_bc', type=float, help='Max Episode Length', default=0.001)
    parser.add_argument('--max_ent_coef_bc', type=float, help='Max Episode Length', default=1.0)
    parser.add_argument('--pretrain_bc_iter', type=int, help='Max Episode Length', default=60001)

    parser.add_argument('--env_id', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--stochastic_exploration', type=str, default="False")
    parser.add_argument('--initial_exploration', type=float, help='Amount of exploration at start', default=1.0)
    parser.add_argument('--stochastic_environment', type=str, choices=['True', 'False'], default='False')
    return parser.parse_args()

class Resize:
    """Resizes and converts RGB Atari frames to grayscale"""

    def __init__(self, frame_height=84, frame_width=84):
        """
        Args:
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
        self.processed = tf.image.rgb_to_grayscale(self.frame)
        self.processed = tf.image.resize_images(self.processed,
                                                [self.frame_height, self.frame_width],
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def process(self, session, frame):
        """
        Args:
            session: A Tensorflow session object
            frame: A (210, 160, 3) frame of an Atari game in RGB
        Returns:
            A processed (96, 96, 1) frame in grayscale
        """
        return session.run(self.processed, feed_dict={self.frame: frame})

class ActionGetter:
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""

    def __init__(self, n_actions, eps_initial=1, eps_final=0.1, eps_final_frame=0.01,
                 eps_evaluation=0.0, eps_annealing_frames=1000000,
                 replay_memory_start_size=50000, max_frames=25000000):
        """
        Args:
            n_actions: Integer, number of possible actions
            eps_initial: Float, Exploration probability for the first
                replay_memory_start_size frames
            eps_final: Float, Exploration probability after
                replay_memory_start_size + eps_annealing_frames frames
            eps_final_frame: Float, Exploration probability after max_frames frames
            eps_evaluation: Float, Exploration probability during evaluation
            eps_annealing_frames: Int, Number of frames over which the
                exploration probabilty is annealed from eps_initial to eps_final
            replay_memory_start_size: Integer, Number of frames during
                which the agent only explores
            max_frames: Integer, Total number of frames shown to the agent
        """
        eps_final = min(eps_initial, eps_final)
        eps_final_frame = min(eps_final, eps_final_frame)
        self.n_actions = n_actions
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.replay_memory_start_size = replay_memory_start_size
        self.max_frames = max_frames

        # Slopes and intercepts for exploration decrease
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.replay_memory_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (
                    self.max_frames - self.eps_annealing_frames - self.replay_memory_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2 * self.max_frames

    def get_eps(self, frame_number):
        if frame_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif frame_number >= self.replay_memory_start_size and frame_number < self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope_2 * frame_number + self.intercept_2
        return eps


    def get_stochastic_action(self, session, state, main_dqn, evaluation=False):
        if evaluation:
            return session.run(main_dqn.best_action, feed_dict={main_dqn.input: [state]})[0]
        else:
            action_prob = session.run(main_dqn.action_prob, feed_dict={main_dqn.input: [state]})[0]
            rand = np.random.uniform(0, 1)
            accumulated_val = 0
            for i in range(action_prob.shape[0]):
                accumulated_val += action_prob[i]
                if rand < accumulated_val:
                    return i


    def get_action(self, session, frame_number, state, main_dqn, evaluation=False):
        """
        Args:
            session: A tensorflow session object
            frame_number: Integer, number of the current frame
            state: A (96, 96, 4) sequence of frames of an Atari game in grayscale
            main_dqn: A DQN object
            evaluation: A boolean saying whether the agent is being evaluated
        Returns:
            An integer between 0 and n_actions - 1 determining the action the agent perfoms next
        """
        if evaluation:
            eps = self.eps_evaluation
        elif frame_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif frame_number >= self.replay_memory_start_size and frame_number < self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope_2 * frame_number + self.intercept_2

        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)
        return session.run(main_dqn.best_action, feed_dict={main_dqn.input: [state]})[0]


class ReplayMemory:
    """Replay Memory that stores the last size=1,000,000 transitions"""

    def __init__(self, size=1000000, frame_height=84, frame_width=84,
                 agent_history_length=4, batch_size=32):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length,self.frame_height, self.frame_width,
                                ), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,self.frame_height, self.frame_width,
                                    ), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward,  terminal):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            frame: A (96, 96, 1) frame of an Atari game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index - self.agent_history_length + 1:index + 1, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        """
        Returns a minibatch of self.batch_size = 32 transitions
        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)

        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[
            self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]


#Note path to csv
def log_data(path, image_path):
    convert_dict = {}
    result_dict = {}
    data = {}
    #try:
    with open(path + 'progress.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            if len(result_dict) == 0:
                for i in range(len(line)):
                    result_dict[i] = line[i]
                    convert_dict[line[i]] = i
                    data[i] = []
            else:
                for i in range(len(line)):
                    data[i].append(float(line[i]))
    plt.plot(data[convert_dict["frame_number"]], data[convert_dict["training_reward"]])
    plt.title("Frame num vs training Rewards")
    plt.xlabel("Frame number")
    plt.ylabel("Training Rewards");
    plt.savefig(image_path + 'training_reward.png')
    plt.close()

    plt.plot(data[convert_dict["frame_number"]], data[convert_dict["td_loss"]])
    plt.title("Frame num vs tdloss")
    plt.xlabel("Frame number")
    plt.ylabel("Tdloss");
    plt.savefig(image_path + 'td_loss.png')
    plt.close()

    if "expert_loss" in convert_dict:
        plt.plot(data[convert_dict["frame_number"]], data[convert_dict["expert_loss"]])
        plt.title("Frame num vs expert loss")
        plt.xlabel("Frame number")
        plt.ylabel("expert loss");
        plt.savefig(image_path + 'expert_loss.png')
        plt.close()

    plt.plot(data[convert_dict["frame_number"]], data[convert_dict["episode_length"]])
    plt.title("Frame num vs Episode Length")
    plt.xlabel("Frame number")
    plt.ylabel("Episode Length");
    plt.savefig(image_path + 'ep_len_loss.png')
    plt.close()
    # except:
    #     print("Figure generation FAILED?")




class TargetNetworkUpdater:
    """Copies the parameters of the main DQN to the target DQN"""

    def __init__(self, main_dqn_vars, target_dqn_vars):
        """
        Args:
            main_dqn_vars: A list of tensorflow variables belonging to the main DQN network
            target_dqn_vars: A list of tensorflow variables belonging to the target DQN network
        """
        self.main_dqn_vars = main_dqn_vars
        self.target_dqn_vars = target_dqn_vars

    def _update_target_vars(self):
        update_ops = []
        for i, var in enumerate(self.main_dqn_vars):
            copy_op = self.target_dqn_vars[i].assign(var.value())
            update_ops.append(copy_op)
        return update_ops

    def update_networks(self, sess):
        """
        Args:
            sess: A Tensorflow session object
        Assigns the values of the parameters of the main network to the
        parameters of the target network
        """
        update_ops = self._update_target_vars()
        for copy_op in update_ops:
            sess.run(copy_op)

def generate_gif(frame_number, frames_for_gif, reward, path):
    """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """
    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3),
                                     preserve_range=True, order=0).astype(np.uint8)

    imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}',
                    frames_for_gif, duration=1 / 30)


class Atari:
    """Wrapper for the environment provided by gym"""

    def __init__(self, envName, stochastic_environment, no_op_steps=10, agent_history_length=4):
        if stochastic_environment == "True":
            self.env = gym.make(envName, frameskip=[2, 6])
        else:
            self.env = gym.make(envName)
        self.frame_processor = Resize()
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.agent_history_length = agent_history_length

    def reset(self, sess, evaluation=False):
        """
        Args:
            sess: A Tensorflow session object
            evaluation: A boolean saying whether the agent is evaluating or training
        Resets the environment and stacks four frames ontop of each other to
        create the first state
        """
        frame = self.env.reset()
        self.last_lives = 0
        terminal_life_lost = True  # Set to true so that the agent starts
        # with a 'FIRE' action when evaluating
        if evaluation:
            for _ in range(random.randint(1, self.no_op_steps)):
                frame, _, _, _ = self.env.step(1)  # Action 'Fire'
        processed_frame = self.frame_processor.process(sess, frame)  # (★★★)
        self.state = np.repeat(processed_frame, self.agent_history_length, axis=2)

        return terminal_life_lost

    def fixed_state(self,sess):
        frame = self.env.reset()
        for _ in range(random.randint(1, self.no_op_steps)):
            frame, _, _, _ = self.env.step(1)  # Action 'Fire'
        processed_frame = self.frame_processor.process(sess, frame)  # (★★★)
        state = np.repeat(processed_frame, self.agent_history_length, axis=2)
        return state

    def step(self, sess, action):
        """
        Args:
            sess: A Tensorflow session object
            action: Integer, action the agent performs
        Performs an action and observes the reward and terminal state from the environment
        """
        new_frame, reward, terminal, info = self.env.step(action)  # (5★)

        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']

        processed_new_frame = self.frame_processor.process(sess, new_frame)  # (6★)
        new_state = np.append(self.state[:, :, 1:], processed_new_frame, axis=2)  # (6★)
        self.state = new_state

        return processed_new_frame, reward, terminal, terminal_life_lost, new_frame

def generate_weights(replay_buff):
    replay_buff.reward_weight = np.zeros((replay_buff.count,))
    max_val = -1
    min_val = 10000000
    prev_index = -1
    accumulated_reward = 0
    for i in range(replay_buff.count):
        accumulated_reward += replay_buff.rewards[i]
        if replay_buff.terminal_flags[i]:
            replay_buff.reward_weight[prev_index+1:i] = accumulated_reward
            if max_val < accumulated_reward:
                max_val = accumulated_reward
            if min_val > accumulated_reward:
                min_val = accumulated_reward
            accumulated_reward = 0
            prev_index = i
    replay_buff.reward_weight = (replay_buff.reward_weight - min_val)/(max_val - min_val)

def get_minibatch(replay_buff):
    """
    Returns a minibatch of self.batch_size = 32 transitions
    """
    if replay_buff.count < replay_buff.agent_history_length:
        raise ValueError('Not enough memories to get a minibatch')

    replay_buff._get_valid_indices()

    for i, idx in enumerate(replay_buff.indices):
        replay_buff.states[i] = replay_buff._get_state(idx - 1)
        replay_buff.new_states[i] = replay_buff._get_state(idx)

    return np.transpose(replay_buff.states, axes=(0, 2, 3, 1)), replay_buff.actions[replay_buff.indices], replay_buff.rewards[
        replay_buff.indices], np.transpose(replay_buff.new_states, axes=(0, 2, 3, 1)), replay_buff.terminal_flags[replay_buff.indices], replay_buff.reward_weight[replay_buff.indices]

def test_q_values(sess, dataset, env, action_getter, dqn, input, output, batch_size, num_expert=40):
    #Test number of states with greater than 75% confidence
    my_replay_memory = ReplayMemory(size=20000, batch_size=batch_size)  # (★)
    env.reset(sess)
    accumulated_rewards = 0
    terminal_life_lost = True
    for i in range(20000):
        action = 1 if terminal_life_lost else action_getter.get_action(sess, 1000,
                                                                        env.state,
                                                                        dqn,
                                                                        evaluation=True)        #print("Action: ",action)
        # (5★)
        processed_new_frame, reward, terminal, terminal_life_lost, _ = env.step(sess, action)
        my_replay_memory.add_experience(action=action,
                                        frame=processed_new_frame[:, :, 0],
                                        reward=reward,
                                        terminal=terminal_life_lost)
        accumulated_rewards += reward
        if terminal:
            break
    count = 0
    #Sample random states
    states_list = []
    for i in range(num_expert):
        states, actions, rewards, new_states, terminal_flags = dataset.get_minibatch()
        states_list.append(states)
    combined_states = np.concatenate(states_list, axis=0)
    values = sess.run(output, feed_dict={input: combined_states})
    for i in range(num_expert * states.shape[0]):
        count += np.max(values[i])
    expert_over_confidence = count/(num_expert * states.shape[0])
    print("Expert Average Max: ", expert_over_confidence)
    env.reset(sess)
    count = 0
    iter = 0
    for i in range(math.ceil(my_replay_memory.count//batch_size) * 5):
        states, actions, rewards, new_states, terminal_flags = my_replay_memory.get_minibatch()
        values = sess.run(output, feed_dict={input: states})
        for j in range(values.shape[0]):
            count += np.max(values[j])
            iter += 1
    generated_over_confidence = count/iter
    print("Generated Average Max: ", generated_over_confidence)
    return generated_over_confidence, expert_over_confidence

def build_initial_replay_buffer(sess, atari, my_replay_memory, action_getter, max_eps, replay_buf_size, MAIN_DQN, args):
    frame_num = 0
    while frame_num < replay_buf_size:
        atari.reset(sess)
        for _ in range(max_eps):
            # (4★)
            if args.stochastic_exploration == "True":
                action = action_getter.get_stochastic_action(sess, atari.state, MAIN_DQN)
            else:
                action = action_getter.get_action(sess, 0, atari.state, MAIN_DQN)
            # print("Action: ",action)
            # (5★)
            processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)
            # (7★) Store transition in the replay memory
            my_replay_memory.add_experience(action=action,
                                            frame=processed_new_frame[:, :, 0],
                                            reward=reward,
                                            terminal=terminal_life_lost)
            frame_num += 1
            if frame_num % (replay_buf_size//10) == 0 and frame_num > 0:
                print(frame_num)
            if terminal:
                break


def sample(args, DQN, name, save=True):
    tf.random.set_random_seed(args.seed)
    tf.reset_default_graph()
    np.random.seed(args.seed)
    # Control parameters
    MAX_EPISODE_LENGTH = args.max_eps_len       # Equivalent of 5 minutes of gameplay at 60 frames per second
    REPLAY_MEMORY_START_SIZE = args.replay_start_size # Number of completely random actions,
                                    # before the agent starts learning
    MAX_FRAMES = args.max_frames            # Total number of frames the agent sees
    NO_OP_STEPS = args.no_op_steps                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an
                                    # evaluation episode
    HIDDEN = args.hidden                    # Number of filters in the final convolutional layer. The output
                                    # has the shape (1,1,1024) which is split into two streams. Both
                                    # the advantage stream and value stream have the shape
                                    # (1,1,512). This is slightly different from the original
                                    # implementation but tests I did with the environment Pong
                                    # have shown that this way the score increases more quickly
    LEARNING_RATE = args.lr         # Set to 0.00025 in Pong for quicker results.
                                    # Hessel et al. 2017 used 0.0000625
    BS = args.batch_size
    atari = Atari(args.env_id, args.stochastic_environment, NO_OP_STEPS)
    atari.env.seed(args.seed)
    # main DQN and target DQN networks:

    with tf.variable_scope('mainDQN'):
        MAIN_DQN = DQN(atari.env.action_space.n, HIDDEN, LEARNING_RATE)  # (★★)
    with tf.variable_scope('targetDQN'):
        TARGET_DQN = DQN(atari.env.action_space.n, HIDDEN)  # (★★)


    init = tf.global_variables_initializer()
    MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
    TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session(config=config)

    action_getter = ActionGetter(atari.env.action_space.n,
                                 replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                 max_frames=MAX_FRAMES,
                                 eps_initial=args.initial_exploration)
    my_replay_memory = ReplayMemory(size=args.num_sampled * MAX_EPISODE_LENGTH, batch_size=BS * 2)
    sess.run(init)
    important_coef_name = ["num_traj", "seed", "lr"]
    important_coef_var = [args.num_sampled, args.seed, args.lr]
    if not args.special_tag == "":
        file_add = args.special_tag + "_"
    else:
        file_add = ""
    for i in range(len(important_coef_name)):
        file_add += important_coef_name[i] + "_" + str(important_coef_var[i]) + "_"
    try:
        if args.checkpoint_file_path != "None":
            saver.restore(sess, args.checkpoint_dir + "/" + name + "/" + args.env_id + "/"  + args.checkpoint_file_path + "model-" + str(args.checkpoint_index))
            print("1. Loaded Model ... ", args.checkpoint_dir + "/" + name + "/" + args.env_id + "/"  + args.checkpoint_file_path + "model-" + str(args.checkpoint_index))
        elif args.checkpoint_index >= 0:
            saver.restore(sess, args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + file_add + "model-" + str(args.checkpoint_index))
            print("2. Loaded Model ... ",  args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + file_add + "model-" + str(args.checkpoint_index))
        else:
            print("3. Model not found ...",  args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + file_add + "model-" + str(args.checkpoint_index))
    except:
        if args.checkpoint_file_path != "None":
            saver.restore(sess, args.checkpoint_dir + "/" + name + "/" + args.env_id + "/"  + args.checkpoint_file_path + "model--" + str(args.checkpoint_index))
            print("1. Loaded Model ... ", args.checkpoint_dir + "/" + name + "/" + args.env_id + "/"  + args.checkpoint_file_path + "model--" + str(args.checkpoint_index))
        elif args.checkpoint_index >= 0:
            saver.restore(sess, args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + file_add + "model--" + str(args.checkpoint_index))
            print("2. Loaded Model ... ",  args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + file_add + "model--" + str(args.checkpoint_index))
        else:
            print("3. Model not found ...",  args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + file_add + "model--" + str(args.checkpoint_index))

    logger.configure(args.log_dir + "/" + name + "/" + args.env_id + "/" + file_add + "/")
    if not os.path.exists(args.gif_dir + "/" + name + "/" + args.env_id + "/"):
        os.makedirs(args.gif_dir + "/" + name + "/" + args.env_id + "/")
    if not os.path.exists(args.checkpoint_dir + "/" + name + "/" + args.env_id + "/"):
        os.makedirs(args.checkpoint_dir + "/" + name + "/" + args.env_id + "/")
    if not os.path.exists(args.expert_dir + "/" + name + "/" + args.env_id + "/"):
        os.makedirs(args.expert_dir + "/" + name + "/" + args.env_id + "/")
    gif = 3
    reward_list = []
    frames = 0
    for _ in tqdm(range(args.num_sampled)):
        terminal_live_lost = atari.reset(sess, evaluation=True)
        episode_reward_sum = 0
        frames_for_gif = []
        for _ in range(MAX_EPISODE_LENGTH):
            action = 1 if terminal_live_lost else action_getter.get_action(sess, 0, atari.state,
                                                                           MAIN_DQN,
                                                                           evaluation=True)
            processed_new_frame, reward, terminal, terminal_live_lost, new_frame = atari.step(sess, action)
            my_replay_memory.add_experience(action=action,
                                            frame=processed_new_frame[:, :, 0],
                                            reward=reward,
                                            terminal=terminal_live_lost)
            episode_reward_sum += reward
            frames += 1
            if terminal_live_lost:
                print(episode_reward_sum, frames)
            if gif > 0:
                frames_for_gif.append(new_frame)
            if terminal == True:
                if gif > 0:
                    try:
                        generate_gif(-gif, frames_for_gif, reward_list[-1], args.gif_dir + "/" + name + "/" + args.env_id + "/" + file_add)
                    except IndexError:
                        print(len(frames_for_gif))
                        print("No evaluation game finished")
                    gif -= 1
                reward_list.append(episode_reward_sum)
                print(len(frames_for_gif))
                break
    print("Average Reward: ", np.mean(reward_list))
    if save:
        pickle.dump(my_replay_memory, open(args.expert_dir + "/" + name + "/" + args.env_id + "/" + args.expert_file + "_" + str(args.num_sampled), "wb"), protocol=4)


def train(args, DQN, learn, name, expert=False, bc_training=None, pretrain_iters=60001):
    MAX_EPISODE_LENGTH = args.max_eps_len       # Equivalent of 5 minutes of gameplay at 60 frames per second
    EVAL_FREQUENCY = args.eval_freq          # Number of frames the agent sees between evaluations
    EVAL_STEPS = args.eval_len               # Number of frames for one evaluation
    NETW_UPDATE_FREQ = args.target_update_freq         # Number of chosen actions between updating the target network.
                                    # According to Mnih et al. 2015 this is measured in the number of
                                    # parameter updates (every four actions), however, in the
                                    # DeepMind code, it is clearly measured in the number
                                    # of actions the agent choses
    DISCOUNT_FACTOR = args.gamma           # gamma in the Bellman equation
    REPLAY_MEMORY_START_SIZE = args.replay_start_size # Number of completely random actions,
                                    # before the agent starts learning
    MAX_FRAMES = args.max_frames            # Total number of frames the agent sees
    MEMORY_SIZE = args.replay_mem_size            # Number of transitions stored in the replay memory
    NO_OP_STEPS = args.no_op_steps                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an
                                    # evaluation episode
    UPDATE_FREQ = args.update_freq                  # Every four actions a gradient descend step is performed
    HIDDEN = args.hidden                    # Number of filters in the final convolutional layer. The output
                                    # has the shape (1,1,1024) which is split into two streams. Both
                                    # the advantage stream and value stream have the shape
                                    # (1,1,512). This is slightly different from the original
                                    # implementation but tests I did with the environment Pong
                                    # have shown that this way the score increases more quickly
    LEARNING_RATE = args.lr         # Set to 0.00025 in Pong for quicker results.
                                    # Hessel et al. 2017 used 0.0000625
    BS = args.batch_size

    atari = Atari(args.env_id, args.stochastic_environment, NO_OP_STEPS)
    atari.env.seed(args.seed)
    # main DQN and target DQN networks:
    if expert:
        with tf.variable_scope('mainDQN'):
            MAIN_DQN = DQN(atari.env.action_space.n, HIDDEN, LEARNING_RATE, bc_learning_rate=args.lr_bc, max_ent_coef=args.max_ent_coef_bc)  # (★★)
        with tf.variable_scope('targetDQN'):
            TARGET_DQN = DQN(atari.env.action_space.n, HIDDEN)  # (★★)
    else:
        with tf.variable_scope('mainDQN'):
            MAIN_DQN = DQN(atari.env.action_space.n, HIDDEN, LEARNING_RATE)  # (★★)
        with tf.variable_scope('targetDQN'):
            TARGET_DQN = DQN(atari.env.action_space.n, HIDDEN)  # (★★)

    init = tf.global_variables_initializer()
    MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
    TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')
    

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    my_replay_memory = ReplayMemory(size=MEMORY_SIZE, batch_size=BS)  # (★)
    network_updater = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
    action_getter = ActionGetter(atari.env.action_space.n,
                                 replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                 max_frames=MAX_FRAMES,
                                 eps_initial=args.initial_exploration)
    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session(config=config)
    sess.run(init)
    fixed_state = np.expand_dims(atari.fixed_state(sess),axis=0)

    important_coef_name = ["num_traj", "seed", "lr", "lr_bc", "max_ent", "stochastic_env"]
    important_coef_var = [args.num_sampled, args.seed, args.lr, args.lr_bc, args.max_ent_coef_bc, args.stochastic_environment]
    if not args.special_tag == "":
        file_add = args.special_tag + "_"
    else:
        file_add = ""
    for i in range(len(important_coef_name)):
        file_add += important_coef_name[i] + "_" + str(important_coef_var[i]) + "_"
    
    if expert:
        if args.expert_file_path != "None":
            dataset = pickle.load(open(args.expert_dir + args.expert_file_path + args.expert_file + "_" + str(args.num_sampled), "rb"))
            print("1. Loaded Data ... ", args.expert_dir + args.expert_file_path + args.expert_file + "_" + str(args.num_sampled))
        else:
            dataset = pickle.load(open(args.expert_dir + "/" + name + "/" + args.env_id + "/" + args.expert_file + "_" + str(args.num_sampled), "rb"))
            print("2. Loaded Data ... ", args.expert_dir + "/" + name + "/" + args.env_id + "/" + args.expert_file + "_" + str(args.num_sampled))
        generate_weights(dataset)
    try:
        if args.checkpoint_file_path != "None":
            saver.restore(sess, args.checkpoint_dir + "/" + name + "/" + args.env_id + "/"  + args.checkpoint_file_path + "model-" + str(args.checkpoint_index))
            print("1. Loaded Model ... ", args.checkpoint_dir + "/" + name + "/" + args.env_id + "/"  + args.checkpoint_file_path + "model-" + str(args.checkpoint_index))
        elif args.checkpoint_index >= 0:
            saver.restore(sess, args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + file_add + "model-" + str(args.checkpoint_index))
            print("2. Loaded Model ... ",  args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + file_add + "model-" + str(args.checkpoint_index))
        else:
            print("3. Model not found ...",  args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + file_add + "model-" + str(args.checkpoint_index))
    except:
        if args.checkpoint_file_path != "None":
            saver.restore(sess, args.checkpoint_dir + "/" + name + "/" + args.env_id + "/"  + args.checkpoint_file_path + "model--" + str(args.checkpoint_index))
            print("1. Loaded Model ... ", args.checkpoint_dir + "/" + name + "/" + args.env_id + "/"  + args.checkpoint_file_path + "model--" + str(args.checkpoint_index))
        elif args.checkpoint_index >= 0:
            saver.restore(sess, args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + file_add + "model--" + str(args.checkpoint_index))
            print("2. Loaded Model ... ",  args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + file_add + "model--" + str(args.checkpoint_index))
        else:
            print("3. Model not found ...",  args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + file_add + "model--" + str(args.checkpoint_index))

    logger.configure(args.log_dir + "/" + name + "/" + args.env_id + "/" + file_add + "/")
    if not os.path.exists(args.gif_dir + "/" + name + "/" + args.env_id + "/"):
        os.makedirs(args.gif_dir + "/" + name + "/" + args.env_id + "/")
    if not os.path.exists(args.checkpoint_dir + "/" + name + "/" + args.env_id + "/"):
        os.makedirs(args.checkpoint_dir + "/" + name + "/" + args.env_id + "/")
    if not os.path.exists(args.expert_dir + "/" + name + "/" + args.env_id + "/"):
        os.makedirs(args.expert_dir + "/" + name + "/" + args.env_id + "/")

    build_initial_replay_buffer(sess, atari, my_replay_memory, action_getter, MAX_EPISODE_LENGTH, REPLAY_MEMORY_START_SIZE, MAIN_DQN, args)
    frame_number = REPLAY_MEMORY_START_SIZE

    if expert and not bc_training is None:
        #Pretrain system .... 
        bc_loss = []
        print("Pretraining ....")
        for i in tqdm(range(pretrain_iters)):
            expert_loss = bc_training(sess, dataset, my_replay_memory, MAIN_DQN)
            bc_loss.append(expert_loss)
            if i % (pretrain_iters//4) == 0 and i > 0:
                print(np.mean(bc_loss[-pretrain_iters//4:]), i)
                test_q_values(sess, dataset, atari, action_getter, MAIN_DQN, MAIN_DQN.input, MAIN_DQN.action_prob_expert, BS)

    eval_rewards = [0]
    rewards = []
    loss_list = []
    bc_loss_list = []
    expert_loss_list = []
    episode_length_list = []
    epoch = 0
    while frame_number < MAX_FRAMES:
        print("Training Model ...")
        epoch_frame = 0
        start_time = time.time()
        while epoch_frame < EVAL_FREQUENCY:
            atari.reset(sess)
            episode_reward_sum = 0
            episode_length = 0
            for _ in range(MAX_EPISODE_LENGTH):
                # (4★)
                if args.stochastic_exploration == "True":
                    action = action_getter.get_stochastic_action(sess, atari.state, MAIN_DQN)
                else:
                    action = action_getter.get_action(sess, frame_number, atari.state, MAIN_DQN)
                #print("Action: ",action)
                # (5★)
                processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)
                frame_number += 1
                epoch_frame += 1
                episode_reward_sum += reward
                episode_length += 1

                # (7★) Store transition in the replay memory
                my_replay_memory.add_experience(action=action,
                                                frame=processed_new_frame[:, :, 0],
                                                reward=reward,
                                                terminal=terminal_life_lost)

                if frame_number % UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                    if expert:
                        bc_loss = bc_training(sess, dataset, my_replay_memory, MAIN_DQN)
                        bc_loss_list.append(bc_loss)
                        loss, expert_loss = learn(sess, dataset, my_replay_memory, MAIN_DQN, TARGET_DQN,
                                    BS, gamma=DISCOUNT_FACTOR)  # (8★)
                        expert_loss_list.append(expert_loss)
                    else:
                        loss = learn(sess, my_replay_memory, MAIN_DQN, TARGET_DQN,
                                    BS, gamma=DISCOUNT_FACTOR)  # (8★)
                    loss_list.append(loss)
                if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                    network_updater.update_networks(sess)  # (9★)

                if terminal:
                    terminal = False
                    break
            rewards.append(episode_reward_sum)
            episode_length_list.append(episode_length)

            # Output the progress:
            if len(rewards) % 25 == 0:
                # logger.log("Runing frame number {0}".format(frame_number))
                logger.record_tabular("frame_number",frame_number)
                logger.record_tabular("training_reward",np.mean(rewards[-100:]))
                if frame_number < REPLAY_MEMORY_START_SIZE:
                    logger.record_tabular("td_loss",0)
                else:
                    logger.record_tabular("td_loss",np.mean(loss_list[-100:]))
                logger.record_tabular("episode_length",np.mean(episode_length_list[-100:]))
                logger.record_tabular("current_exploration", action_getter.get_eps(frame_number))
                logger.record_tabular("evaluation_reward", np.mean(eval_rewards))
                q_vals = sess.run(MAIN_DQN.q_values, feed_dict={MAIN_DQN.input: fixed_state})
                for i in range(atari.env.action_space.n):
                    logger.record_tabular("q_val_action {0}".format(i),q_vals[0,i])
                action_probs = sess.run(MAIN_DQN.action_prob, feed_dict={MAIN_DQN.input: fixed_state})
                for i in range(atari.env.action_space.n):
                    logger.record_tabular("action_prob {0}".format(i),action_probs[0,i])
                if expert:
                    if frame_number < REPLAY_MEMORY_START_SIZE:
                        logger.record_tabular("expert_loss", 0)
                        logger.record_tabular("generated_overconfidence:", 0)
                        logger.record_tabular("expert_overconfidence:", 0)
                        logger.record_tabular("bc_loss", 0)
                    else:
                        logger.record_tabular("expert_loss", np.mean(expert_loss_list[-100:]))
                        generated_overconfidence, expert_overconfidence = test_q_values(sess, dataset, atari, action_getter, MAIN_DQN, MAIN_DQN.input, MAIN_DQN.action_prob_expert, BS)
                        logger.record_tabular("generated_overconfidence:", generated_overconfidence)
                        logger.record_tabular("expert_overconfidence:", expert_overconfidence)
                        logger.record_tabular("bc_loss", np.mean(bc_loss_list[-100:]))
                print("Completion: ", str(epoch_frame)+"/"+str(EVAL_FREQUENCY))
                print("Current Frame: ",frame_number)
                logger.dumpkvs()
        #Evaluation ...
        gif = True
        frames_for_gif = []
        eval_rewards = []
        evaluate_frame_number = 0
        print("Evaluating Model.... ")
        while evaluate_frame_number < EVAL_STEPS:
            terminal_life_lost = atari.reset(sess, evaluation=True)
            episode_reward_sum = 0
            for _ in range(MAX_EPISODE_LENGTH):
                # Fire (action 1), when a life was lost or the game just started,
                # so that the agent does not stand around doing nothing. When playing
                # with other environments, you might want to change this...
                if terminal_life_lost:
                    action = 1
                else:
                    action = action_getter.get_action(sess, frame_number,
                                                            atari.state,
                                                            MAIN_DQN,
                                                            evaluation=True)
                processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
                evaluate_frame_number += 1
                episode_reward_sum += reward
                if gif:
                    frames_for_gif.append(new_frame)
                if terminal:
                    eval_rewards.append(episode_reward_sum)
                    gif = False  # Save only the first game of the evaluation as a gif
                    break
            if len(eval_rewards) % 10 == 0:
                print("Evaluation Completion: ", str(evaluate_frame_number) + "/" + str(EVAL_STEPS))
        print("\n\n\n-------------------------------------------")
        print("Evaluation score:\n", np.mean(eval_rewards))
        print("-------------------------------------------\n\n\n")
        try:
            generate_gif(frame_number, frames_for_gif, eval_rewards[0],
                              args.gif_dir + "/" + name + "/" + args.env_id + "/" + file_add)
        except IndexError:
            print("No evaluation game finished")
        saver.save(sess, args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + file_add + "model",
                global_step=frame_number)

        # log_data(args.log_dir + "/" + name + "/" + args.env_id + "/" + file_add + "/",
        #                args.log_dir + "/" +  name + "/" + args.env_id + "/" + file_add + "/")
        # Save the network parameters
        print("Runtime: ", time.time() - start_time)
        print("Epoch: ", epoch, "Total Frames: ", frame_number)
        epoch += 1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          