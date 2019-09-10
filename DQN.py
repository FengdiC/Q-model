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
        self.conv4 = tf.layers.conv2d(
            inputs=self.conv3, filters=hidden, kernel_size=[7, 7], strides=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4')
        self.d = tf.layers.flatten(self.conv4)
        self.dense = tf.layers.dense(inputs = self.d,units = 1024,
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
    imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}',frames_for_gif, duration=1 / 30)

class Atari:
    """Wrapper for the environment provided by gym"""

    def __init__(self, envName, no_op_steps=10, agent_history_length=4):
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

import argparse
def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of DQN")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_dir', type=str, default='dqn_data/')
    parser.add_argument('--expert_file', type=str, default='expert_data.pkl')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='models/vanilla_dqn/')
    parser.add_argument('--checkpoint_index', type=int, help='index of model to load', default=-1)
    parser.add_argument('--log_dir', help='the directory to save log file', default='logs/vanilla_dqn/')
    parser.add_argument('--gif_dir', help='the directory to save GIFs file', default='GIFs/vanilla_dqn/')
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    parser.add_argument('--num_sampled', type=int, help='Num Generated Sequence', default=1)
    parser.add_argument('--max_eps_len', type=int, help='Max Episode Length', default=18000)
    parser.add_argument('--eval_freq', type=int, help='Evaluation Frequency', default=200000)
    parser.add_argument('--eval_len', type=int, help='Max Episode Length', default=10000)
    parser.add_argument('--target_update_freq', type=int, help='Max Episode Length', default=10000)
    parser.add_argument('--replay_start_size', type=int, help='Max Episode Length', default=50000)
    parser.add_argument('--max_frames', type=int, help='Max Episode Length', default=3000000)
    parser.add_argument('--replay_mem_size', type=int, help='Max Episode Length', default=1000000)
    parser.add_argument('--no_op_steps', type=int, help='Max Episode Length', default=10)
    parser.add_argument('--update_freq', type=int, help='Max Episode Length', default=4)
    parser.add_argument('--hidden', type=int, help='Max Episode Length', default=1024)
    parser.add_argument('--batch_size', type=int, help='Max Episode Length', default=32)
    parser.add_argument('--gamma', type=float, help='Max Episode Length', default=0.99)
    parser.add_argument('--lr', type=float, help='Max Episode Length', default=0.0000625)
    parser.add_argument('--env_id', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--initial_exploration', type=float, help='Amount of exploration at start', default=1.0)
    return parser.parse_args()

args = argsparser()
tf.reset_default_graph()
# Control parameters
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
atari = Atari(args.env_id, NO_OP_STEPS)
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

    my_replay_memory = ReplayMemory(size=MEMORY_SIZE, batch_size=BS)  # (★)
    network_updater = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
    action_getter = ActionGetter(atari.env.action_space.n,
                                 replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                 max_frames=MAX_FRAMES,
                                 eps_initial=args.initial_exploration)

    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session(config=config)
    sess.run(init)
    if args.checkpoint_index >= 0:
        saver.restore(sess, args.checkpoint_dir + "model--" + str(args.checkpoint_index))
        print("Loaded Model ... ", args.checkpoint_dir + "model--" + str(args.checkpoint_index))
    logger.configure(args.log_dir)

    fixed_state = np.expand_dims(atari.fixed_state(sess),axis=0)
    frame_number = 0
    rewards = []
    loss_list = []

    if not os.path.exists(args.gif_dir):
        os.makedirs(args.gif_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    while frame_number < MAX_FRAMES:
        print("Training Model ...")
        epoch_frame = 0
        start_time = time.time()
        while epoch_frame < EVAL_FREQUENCY:
            atari.reset(sess)
            episode_reward_sum = 0
            for _ in range(MAX_EPISODE_LENGTH):
                # (4★)
                action = action_getter.get_action(sess, frame_number, atari.state, MAIN_DQN)
                #print("Action: ",action)
                # (5★)
                processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)
                frame_number += 1
                epoch_frame += 1
                episode_reward_sum += reward

                # (7★) Store transition in the replay memory
                my_replay_memory.add_experience(action=action,
                                                frame=processed_new_frame[:, :, 0],
                                                reward=reward,
                                                terminal=terminal_life_lost)

                if frame_number % UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                    loss = learn(sess, my_replay_memory, MAIN_DQN, TARGET_DQN,
                                 BS, gamma=DISCOUNT_FACTOR)  # (8★)
                    loss_list.append(loss)
                if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                    network_updater.update_networks(sess)  # (9★)

                if terminal:
                    terminal = False
                    break
            rewards.append(episode_reward_sum)

            # Output the progress:
            if len(rewards) % 10 == 0:
                if frame_number > REPLAY_MEMORY_START_SIZE:
                    loss_list = []
                q_vals = sess.run(MAIN_DQN.q_values, feed_dict={MAIN_DQN.input: fixed_state})
                # logger.log("Runing frame number {0}".format(frame_number))
                logger.record_tabular("frame_number",frame_number)
                logger.record_tabular("training_reward",np.mean(rewards[-100:]))
                for i in range(atari.env.action_space.n):
                    logger.record_tabular("q_val action {0}".format(i),q_vals[0,i])
                print("Completion: ", str(epoch_frame)+"/"+str(EVAL_FREQUENCY))
                print("Current Frame: ",frame_number)
                print("Average Reward: ", np.mean(rewards[-100:]))
                if frame_number > REPLAY_MEMORY_START_SIZE:
                    print("Average Loss: ", np.mean(loss_list[-100:]))

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
                action = 1 if terminal_life_lost else action_getter.get_action(sess, frame_number,
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
            generate_gif(frame_number, frames_for_gif, eval_rewards[0], args.gif_dir)
        except IndexError:
            print("No evaluation game finished")
        logger.log("Average Evaluation Reward", np.mean(eval_rewards))
        logger.log("Average Sequence Length", evaluate_frame_number/len(eval_rewards))
        # Save the network parameters
        saver.save(sess, args.checkpoint_dir + 'model-', global_step=frame_number)
        print("Runtime: ", time.time() - start_time)
        logger.dumpkvs()

def test():
    action_getter = ActionGetter(atari.env.action_space.n,
                                 replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                 max_frames=MAX_FRAMES)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if args.checkpoint_index >= 0:
        saver = tf.train.import_meta_graph(args.checkpoint_dir + "model--" + str(args.checkpoint_index) + ".meta")
        saver.restore(sess, args.checkpoint_dir + "model--" + str(args.checkpoint_index))
        print("Loaded Model ... ")
    frames_for_gif = []
    terminal_live_lost = atari.reset(sess, evaluation=True)
    episode_reward_sum = 0
    obs = np.zeros((10,500,84,84,4))
    acs = np.zeros((10,500,1))
    ep_rets = np.zeros(10)
    rets = 0
    gamma = 0.99
    while True:
        atari.env.render()
        action = 1 if terminal_live_lost else action_getter.get_action(sess, 0, atari.state,
                                                                       MAIN_DQN,
                                                                       evaluation=True)
        processed_new_frame, reward, terminal, terminal_live_lost, new_frame = atari.step(sess, action)
        episode_reward_sum += reward
        frames_for_gif.append(new_frame)
        if terminal == True:
            break

    atari.env.close()
    for i in range(10):
        terminal_live_lost = atari.reset(sess, evaluation=True)
        for j in range(500):
            obs[i, j] = atari.state
            action = 1 if terminal_live_lost else action_getter.get_action(sess, 0, atari.state,
                                                                           MAIN_DQN,
                                                                           evaluation=True)
            acs[i,j] = action
            processed_new_frame, reward, terminal, terminal_live_lost, new_frame = atari.step(sess, action)
            rets = reward + gamma * rets
            if terminal:
                print("The game ends in 500 steps")
        ep_rets[i] = rets
        rets =0

    np.savez('deterministic.dqn.breakout.npz', obs=obs, acs=acs, ep_rets=ep_rets)
    print("The total reward is {}".format(episode_reward_sum))
    print("Creating gif...")
    generate_gif(0, frames_for_gif, episode_reward_sum, PATH)

def sample(args):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session(config=config)
    if not os.path.exists(args.expert_dir):
        os.makedirs(args.expert_dir)
    action_getter = ActionGetter(atari.env.action_space.n,
                                 replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                 max_frames=MAX_FRAMES)
    my_replay_memory = ReplayMemory(size=args.num_sampled * MAX_EPISODE_LENGTH, batch_size=BS)
    sess.run(init)
    if args.checkpoint_index >= 0:
        saver.restore(sess, args.checkpoint_dir + "model--" + str(args.checkpoint_index))
        print("Loaded Model ... ")
    gif = True
    frames_for_gif = []
    reward_list = []
    for _ in tqdm(range(args.num_sampled)):
        terminal_live_lost = atari.reset(sess, evaluation=True)
        episode_reward_sum = 0
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
            if gif:
                frames_for_gif.append(new_frame)
            if terminal == True:
                gif = False
                reward_list.append(episode_reward_sum)
                break
    try:
        generate_gif(-1, frames_for_gif, reward_list[0], args.gif_dir)
    except IndexError:
        print("No evaluation game finished")
    print("Average Reward: ", np.mean(reward_list))
    pickle.dump(my_replay_memory, open(args.expert_dir + args.expert_file + "_" + str(args.num_sampled), "wb"))

tf.random.set_random_seed(args.seed)
if args.task == "train":
    train(args)
elif args.task == "evaluate":
    test(args)
else:
    sample(args)