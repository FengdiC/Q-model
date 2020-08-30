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
from tensorboard_logger import tensorflowboard_logger

class DQN:
    """Implements a Deep Q Network"""

    def __init__(self, args, n_actions=4, hidden=1024,
                 frame_height=84, frame_width=84, agent_history_length=4, name="dqn"):
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
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length

        self.input = tf.placeholder(shape=[None, self.frame_height,
                                           self.frame_width, self.agent_history_length],
                                    dtype=tf.float32)
        # Normalizing the input
        self.inputscaled = (self.input - 127.5)/127.5
        #self.inputscaled = self.input
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
        self.loss_per_sample = tf.losses.huber_loss(labels=self.target_q, predictions=self.Q, reduction=tf.losses.Reduction.NONE)
        self.loss = tf.reduce_mean(self.loss_per_sample)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        self.update = self.optimizer.minimize(self.loss)

def learn(session, states, actions, diffs, rewards, new_states, terminal_flags, main_dqn, target_dqn, batch_size, gamma, args):
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
    #print(np.max(states), actions, rewards)
    # Draw a minibatch from the replay memory
    # states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch()
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
    loss, q_val, q_values, _ = session.run([main_dqn.loss_per_sample, main_dqn.Q, main_dqn.q_values, main_dqn.update],
                          feed_dict={main_dqn.input:states,
                                     main_dqn.target_q:target_q,
                                     main_dqn.action:actions})
    # for i in range(batch_size):
    #     if loss[i] > 5:
    #         print(i, loss[i], q_val[i], target_q[i], rewards[i], terminal_flags[i])
    # if np.sum(terminal_flags) > 0:
    #     quit()
    return loss
    
def train(name="dqn", priority=True):
    args = utils.argsparser()
    tf.random.set_random_seed(args.seed)
    np.random.seed(args.seed)

    
    MAX_EPISODE_LENGTH = args.max_eps_len       # Equivalent of 5 minutes of gameplay at 60 frames per second
    EVAL_FREQUENCY = args.eval_freq          # Number of frames the agent sees between evaluations
    GIF_FREQUENCY = args.gif_freq
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
        MAIN_DQN = DQN(args, atari.env.action_space.n, HIDDEN, name="mainDQN")  #
    with tf.variable_scope('targetDQN'):
        TARGET_DQN = DQN(args, atari.env.action_space.n, HIDDEN, name="targetDQN")  #

    init = tf.global_variables_initializer()
    MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
    TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if priority:
        my_replay_memory = PriorityBuffer.PrioritizedReplayBuffer(MEMORY_SIZE, args.alpha, args.var)  #
    else:
        my_replay_memory = PriorityBuffer.ReplayBuffer(MEMORY_SIZE)  #

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

    if not os.path.exists("./" + args.gif_dir + "/" + name + "/" + args.env_id + "/"):
        os.makedirs("./" + args.gif_dir + "/" + name + "/" + args.env_id + "/")
    if not os.path.exists("./" + args.checkpoint_dir + "/" + name + "/" + args.env_id + "/"):
        os.makedirs("./" + args.checkpoint_dir + "/" + name + "/" + args.env_id + "/")
    if os.path.exists(args.expert_dir + args.expert_file):
        my_replay_memory.load_expert_data( args.expert_dir + args.expert_file)
    else:
        print("No Expert Data ... ")
    #utils.train_step(sess, args, MAIN_DQN, TARGET_DQN, network_updater, action_getter, my_replay_memory, atari, 0, args.pretrain_bc_iter, learn, pretrain=True, priority=False)
    tflogger = tensorflowboard_logger("./" + args.log_dir + "/" + name + "_" + args.env_id + "_priority_" + str(priority), sess, args)
    utils.build_initial_replay_buffer(sess, atari, my_replay_memory, action_getter, MAX_EPISODE_LENGTH, REPLAY_MEMORY_START_SIZE, MAIN_DQN, args)
    eval_reward, eval_var = utils.evaluate_model(sess, args, EVAL_STEPS, MAIN_DQN, action_getter, MAX_EPISODE_LENGTH, atari, frame_number, model_name=name, gif=True, random=False)
    tflogger.log_scalar("Evaluation/Reward", eval_reward, frame_number)
    tflogger.log_scalar("Evaluation/Reward Variance", eval_var, frame_number)

    last_eval = 0
    last_gif = 0
    while frame_number < MAX_FRAMES:
        eps_rw, eps_len, eps_loss, eps_time, exp_ratio = utils.train_step(sess, args, MAIN_DQN, TARGET_DQN, network_updater, action_getter, my_replay_memory, atari, frame_number, MAX_EPISODE_LENGTH, learn, priority=priority, pretrain=False)
        frame_number += eps_len
        eps_number += 1
        last_gif += 1
        last_eval += eps_len

        tflogger.log_scalar("Episode/Reward", eps_rw, frame_number)
        tflogger.log_scalar("Episode/Length", eps_len, frame_number)
        tflogger.log_scalar("Episode/Loss/Total", eps_loss, frame_number)
        tflogger.log_scalar("Episode/Time", eps_time, frame_number)
        tflogger.log_scalar("Episode/Reward Per Frame", eps_rw / max(1, eps_len),
                            frame_number)
        tflogger.log_scalar("Episode/Expert Ratio", exp_ratio, frame_number)
        tflogger.log_scalar("Episode/Exploration", action_getter.get_eps(frame_number), frame_number)
        tflogger.log_scalar("Total Episodes", eps_number, frame_number)
        tflogger.log_scalar("Replay Buffer Size", my_replay_memory.count, frame_number)

        if EVAL_FREQUENCY <= last_eval:
            last_eval = last_eval - EVAL_FREQUENCY
            if GIF_FREQUENCY <= last_gif:
                last_gif = last_gif - GIF_FREQUENCY
                eval_reward, eval_var = utils.evaluate_model(sess, args, EVAL_STEPS, MAIN_DQN, action_getter, MAX_EPISODE_LENGTH, atari,
                                 frame_number, model_name=name, gif=True)
            else:
                eval_reward, eval_var = utils.evaluate_model(sess, args, EVAL_STEPS, MAIN_DQN, action_getter, MAX_EPISODE_LENGTH, atari,
                                 frame_number, model_name=name, gif=False)
            tflogger.log_scalar("Evaluation/Reward", eval_reward, frame_number)
            tflogger.log_scalar("Evaluation/Reward Variance", eval_var, frame_number)
            saver.save(sess, "./" + args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + "model",
                    global_step=frame_number)
train()
