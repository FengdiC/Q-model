import tensorflow as tf
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
                 frame_height=84, frame_width=84, agent_history_length=4, max_ent_coefficient=1.0):
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
        self.dense = tf.layers.dense(inputs=self.d, units=512,
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2), name="fc5")

        # Splitting into value and advantage stream
        self.valuestream, self.advantagestream = tf.split(self.dense, 2, -1)
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
        self.best_action = tf.argmax(self.q_values, 1)
        self.action_prob = tf.nn.softmax(self.q_values)
        self.action_prob_expert = self.action_prob

        self.expert_action = tf.placeholder(shape=[None], dtype=tf.int32)
        self.expert_weights = tf.placeholder(shape=[None], dtype=tf.float32)
        self.prob = tf.reduce_sum(tf.multiply(self.action_prob,
                                              tf.one_hot(self.expert_action, self.n_actions, dtype=tf.float32)),
                                  axis=1)
        self.behavior_cloning_loss = tf.reduce_mean(-tf.log(self.prob + 0.00001) * self.expert_weights)  # + a l2 reg to prevent overtraining too much
        #self.behavior_cloning_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.q_values, labels=tf.one_hot(self.expert_action, self.n_actions, dtype=tf.float32)))
        self.regularization = tf.reduce_mean(
            tf.reduce_sum(self.action_prob * tf.log(self.action_prob + 0.000001), axis=1))
        self.behavior_cloning_loss += max_ent_coefficient #* self.regularization

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.behavior_cloning_loss)


def learn(session, dataset, main_dqn, target_dqn, batch_size, gamma):
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
    # max_val = np.max(dataset.reward_per_mini_batch)
    states, actions, _, _, _, weights = utils.get_minibatch(dataset)
    # weights = np.copy(weights)/max_val
    loss, _ = session.run([main_dqn.behavior_cloning_loss, main_dqn.update],
                          feed_dict={main_dqn.input: states,
                                     main_dqn.expert_action: actions,
                                     main_dqn.expert_weights: weights})
    return loss


import argparse


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of DQN")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_dir', type=str, default='dqn_data/')
    parser.add_argument('--expert_file', type=str, default='expert_data.pkl')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='models/bc/')
    parser.add_argument('--checkpoint_index', type=int, help='index of model to load', default=-1)
    parser.add_argument('--log_dir', help='the directory to save log file', default='logs/bc/')
    parser.add_argument('--gif_dir', help='the directory to save GIFs file', default='GIFs/bc/')
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    parser.add_argument('--num_sampled', type=int, help='Num Generated Sequence', default=1)
    parser.add_argument('--max_eps_len', type=int, help='Max Episode Length', default=18000)
    parser.add_argument('--eval_freq', type=int, help='Evaluation Frequency', default=100000)
    parser.add_argument('--eval_len', type=int, help='Max Episode Length', default=10000)
    parser.add_argument('--target_update_freq', type=int, help='Max Episode Length', default=10000)
    parser.add_argument('--replay_start_size', type=int, help='Max Episode Length', default=50000)
    parser.add_argument('--max_frames', type=int, help='Max Episode Length', default=3000000)
    parser.add_argument('--replay_mem_size', type=int, help='Max Episode Length', default=1000000)
    parser.add_argument('--no_op_steps', type=int, help='Max Episode Length', default=10)
    parser.add_argument('--update_freq', type=int, help='Max Episode Length', default=4)
    parser.add_argument('--hidden', type=int, help='Max Episode Length', default=512)
    parser.add_argument('--batch_size', type=int, help='Max Episode Length', default=32)
    parser.add_argument('--gamma', type=float, help='Max Episode Length', default=0.99)
    parser.add_argument('--lr', type=float, help='Max Episode Length', default=0.001)
    parser.add_argument('--env_id', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--initial_exploration', type=float, help='Amount of exploration at start', default=1.0)
    parser.add_argument('--stochastic_environment', type=str, choices=['True', 'False'], default='False')
    return parser.parse_args()


args = argsparser()
tf.random.set_random_seed(args.seed)
np.random.seed(args.seed)

tf.reset_default_graph()
# Control parameters
MAX_EPISODE_LENGTH = args.max_eps_len  # Equivalent of 5 minutes of gameplay at 60 frames per second
EVAL_FREQUENCY = args.eval_freq  # Number of frames the agent sees between evaluations
EVAL_STEPS = args.eval_len  # Number of frames for one evaluation
NETW_UPDATE_FREQ = args.target_update_freq  # Number of chosen actions between updating the target network.
# According to Mnih et al. 2015 this is measured in the number of
# parameter updates (every four actions), however, in the
# DeepMind code, it is clearly measured in the number
# of actions the agent choses
DISCOUNT_FACTOR = args.gamma  # gamma in the Bellman equation
REPLAY_MEMORY_START_SIZE = args.replay_start_size  # Number of completely random actions,
# before the agent starts learning
MAX_FRAMES = args.max_frames  # Total number of frames the agent sees
MEMORY_SIZE = args.replay_mem_size  # Number of transitions stored in the replay memory
NO_OP_STEPS = args.no_op_steps  # Number of 'NOOP' or 'FIRE' actions at the beginning of an
# evaluation episode
UPDATE_FREQ = args.update_freq  # Every four actions a gradient descend step is performed
HIDDEN = args.hidden  # Number of filters in the final convolutional layer. The output
# has the shape (1,1,1024) which is split into two streams. Both
# the advantage stream and value stream have the shape
# (1,1,512). This is slightly different from the original
# implementation but tests I did with the environment Pong
# have shown that this way the score increases more quickly
LEARNING_RATE = args.lr  # Set to 0.00025 in Pong for quicker results.
# Hessel et al. 2017 used 0.0000625
BS = args.batch_size
atari = utils.Atari(args.env_id, args.stochastic_environment, NO_OP_STEPS)
atari.env.seed(args.seed)
print("The environment has the following {} actions: {}".format(atari.env.action_space.n,
                                                                atari.env.unwrapped.get_action_meanings()))

# main DQN and target DQN networks:
with tf.variable_scope('mainDQN'):
    MAIN_DQN = DQN(atari.env.action_space.n, HIDDEN, LEARNING_RATE)  # (★★)
with tf.variable_scope('targetDQN'):
    TARGET_DQN = DQN(atari.env.action_space.n, HIDDEN)  # (★★)

init = tf.global_variables_initializer()
MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')


def train(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    dataset = pickle.load(open("data/" + args.expert_file + "_" + str(args.num_sampled), "rb"))
    dataset.min_reward = 0
    dataset.max_reward = 1
    action_getter = utils.ActionGetter(atari.env.action_space.n,
                                       replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                       max_frames=MAX_FRAMES,
                                       eps_initial=args.initial_exploration)

    utils.generate_weights(dataset)
    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session(config=config)
    sess.run(init)
    fixed_state = np.expand_dims(atari.fixed_state(sess), axis=0)

    if args.checkpoint_index >= 0:
        saver.restore(sess, args.checkpoint_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/" + "model--" + str(
            args.checkpoint_index))
        print("Loaded Model ... ", args.checkpoint_dir + args.env_id + "seed_" + str(args.seed) + "/" + "model--" + str(
            args.checkpoint_index))
    logger.configure(args.log_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/")
    if not os.path.exists(args.gif_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/"):
        os.makedirs(args.gif_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/")
    if not os.path.exists(args.checkpoint_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/"):
        os.makedirs(args.checkpoint_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/")

    frame_number = 0
    loss_list = []
    epoch = 0
    while frame_number < MAX_FRAMES:
        print("Training Model ...")
        epoch_frame = 0
        start_time = time.time()
        for j in tqdm(range(EVAL_FREQUENCY // BS)):
            loss = learn(sess, dataset, MAIN_DQN, TARGET_DQN, BS, gamma=DISCOUNT_FACTOR)  # (8★)
            loss_list.append(loss)
            # Output the progress:
        # logger.log("Runing frame number {0}".format(frame_number))
        logger.record_tabular("frame_number", frame_number)
        logger.record_tabular("td loss", np.mean(loss_list[-100:]))
        q_vals = sess.run(MAIN_DQN.action_prob, feed_dict={MAIN_DQN.input: fixed_state})
        for i in range(atari.env.action_space.n):
            logger.record_tabular("q_val action {0}".format(i), q_vals[0, i])
        utils.test_q_values(sess, dataset, atari, action_getter, MAIN_DQN, MAIN_DQN.input, MAIN_DQN.action_prob_expert,
                            BS)
        print("Current Frame: ", frame_number)
        print("TD Loss: ", np.mean(loss_list[-100:]))

        # Evaluation ...
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
                action = 1 if terminal_life_lost and args.env_id == "BreakoutDeterministic-v4" else action_getter.get_action(sess, frame_number,
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
        print("Evaluation score:\n", np.mean(eval_rewards))
        try:
            utils.generate_gif(frame_number, frames_for_gif, eval_rewards[0],
                               args.gif_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/")
        except IndexError:
            print("No evaluation game finished")
        logger.log("Average Evaluation Reward", np.mean(eval_rewards))
        logger.log("Average Sequence Length", evaluate_frame_number / len(eval_rewards))
        # Save the network parameters
        saver.save(sess, args.checkpoint_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/" + 'model-',
                   global_step=frame_number)
        print("Runtime: ", time.time() - start_time)
        print("Epoch: ", epoch, "Total Frames: ", frame_number)
        epoch += 1
        logger.dumpkvs()


if args.task == "train":
    train(args)
elif args.task == "evaluate":
    utils.sample(args, DQN, save=False)
else:
    utils.sample(args, DQN)