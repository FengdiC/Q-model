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
import math
import time
import pickle
from numpy import genfromtxt
import csv

import argparse
def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of DQN")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_dir', type=str, default='Q-model/data/')
    parser.add_argument('--expert_file', type=str, default='expert_data.pkl')
    parser.add_argument('--expert_file_path', type=str, default='None')

    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='Q-model/models/')
    parser.add_argument('--checkpoint_index', type=int, help='index of model to load', default=-1)
    parser.add_argument('--checkpoint_file_path', type=str, default='None')
    parser.add_argument('--special_tag', type=str, default='')

    parser.add_argument('--log_dir', help='the directory to save log file', default='Q-model/logs/')
    parser.add_argument('--gif_dir', help='the directory to save GIFs file', default='Q-model/gifs/')
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
    parser.add_argument('--hidden', type=int, help='Max Episode Length', default=512)
    parser.add_argument('--batch_size', type=int, help='Max Episode Length', default=32)
    parser.add_argument('--gamma', type=float, help='Max Episode Length', default=0.99)
    parser.add_argument('--lr', type=float, help='Max Episode Length', default=0.0000625)
    parser.add_argument('--lr_bc', type=float, help='Max Episode Length', default=0.0001)
    parser.add_argument('--lr_expert', type=float, help='Max Episode Length', default=0.00001)
    parser.add_argument('--td_iterations', type=int, help='Max Episode Length', default=1)
    parser.add_argument('--expert_iterations', type=int, help='Max Episode Length', default=1)


    parser.add_argument('--decay_rate', type=int, help='Max Episode Length', default=1000000)
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

    def get_bc_action(self, session, state, main_dqn):
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
        action_prob = session.run(main_dqn.action_prob_expert, feed_dict={main_dqn.input: [state]})[0]
        rand = np.random.uniform(0, 1)
        accumulated_val = 0
        for i in range(action_prob.shape[0]):
            accumulated_val += action_prob[i]
            if rand < accumulated_val:
                return i


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
        self.max_reward = 1
        self.min_reward = 0
        self.total_count = 0

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
        if reward > self.max_reward:
            self.max_reward = reward
        if reward < self.min_reward:
            self.min_reward = reward
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size
        self.total_count += 1

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

        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], (self.rewards[
            self.indices] - self.min_reward)/(self.max_reward - self.min_reward), np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]


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
    plt.plot(data[convert_dict["frame_number"]], data[convert_dict["evaluation_reward"]])
    plt.title("Frame num vs training Rewards")
    plt.xlabel("Frame number")
    plt.ylabel("Rewards");
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
    replay_buff.reward_weight = np.exp((replay_buff.reward_weight - max_val)* 0.1)
    #ln(x) = ln(exp(weight)) - ln(exp(max))
    #e^(ln(exp(weight)) - ln(exp(max))) = x

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
            
            action = 1 if terminal_live_lost and args.env_id == "BreakoutDeterministic-v4" else action_getter.get_action(sess, 0, atari.state,
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


def train(args, DQN, learn, name, expert=False, bc_training=None, pretrain_iters=60001, only_pretrain=True):
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
            MAIN_DQN = DQN(args, atari.env.action_space.n, HIDDEN, max_ent_coef=args.max_ent_coef_bc)  # (★★)
        with tf.variable_scope('targetDQN'):
            TARGET_DQN = DQN(args, atari.env.action_space.n, HIDDEN)  # (★★)
    else:
        with tf.variable_scope('mainDQN'):
            MAIN_DQN = DQN(args, atari.env.action_space.n, HIDDEN)  # (★★)
        with tf.variable_scope('targetDQN'):
            TARGET_DQN = DQN(args, atari.env.action_space.n, HIDDEN)  # (★★)

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
        dataset.min_reward = 0
        dataset.max_reward = 1
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
        for i in range(pretrain_iters):
            expert_loss = bc_training(sess, dataset, my_replay_memory, MAIN_DQN)
            bc_loss.append(expert_loss)
            if i % (pretrain_iters//8) == 0 and i > 0:
                print(np.mean(bc_loss[-100:]), i)
                test_q_values(sess, dataset, atari, action_getter, MAIN_DQN, MAIN_DQN.input, MAIN_DQN.action_prob_expert, BS)
                print("Evaluating Model.... ")
                evaluate_frame_number = 0
                frames_for_gif = []
                eval_rewards = []
                gif = True
                while evaluate_frame_number < EVAL_STEPS:
                    terminal_life_lost = atari.reset(sess, evaluation=True)
                    episode_reward_sum = 0
                    for _ in range(MAX_EPISODE_LENGTH):
                        # Fire (action 1), when a life was lost or the game just started,
                        # so that the agent does not stand around doing nothing. When playing
                        # with other environments, you might want to change this...
                        if terminal_life_lost and False:
                            action = 1
                        else:
                            action = action_getter.get_bc_action(sess,
                                                              atari.state,
                                                              MAIN_DQN)
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
    eval_rewards = [0]
    rewards = []
    loss_list = []
    bc_loss_list = []
    expert_loss_list = []
    episode_length_list = []
    epoch = 0

    gif = True
    frames_for_gif = []
    eval_rewards = []
    evaluate_frame_number = 0
    print("Evaluating Pretrained Model.... ")
    while evaluate_frame_number < EVAL_STEPS:
        terminal_life_lost = atari.reset(sess, evaluation=True)
        episode_reward_sum = 0
        for _ in range(MAX_EPISODE_LENGTH):
            # Fire (action 1), when a life was lost or the game just started,
            # so that the agent does not stand around doing nothing. When playing
            # with other environments, you might want to change this...
            if terminal_life_lost and args.env_id == "BreakoutDeterministic-v4":
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
                    if expert and frame_number > REPLAY_MEMORY_START_SIZE*5:
                        # bc_loss = bc_training(sess, dataset, my_replay_memory, MAIN_DQN)
                        # bc_loss_list.append(bc_loss)
                        loss, expert_loss = learn(sess, dataset, my_replay_memory, MAIN_DQN, TARGET_DQN,
                                    BS, DISCOUNT_FACTOR, args)  # (8★)
                        expert_loss_list.append(expert_loss)
                    else:
                        loss, expert_loss = learn(sess, dataset, my_replay_memory, MAIN_DQN, TARGET_DQN,
                                    BS, DISCOUNT_FACTOR, args,no_expert=True)  # (8★)  # (8★)
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
                logger.record_tabular("max_reward", my_replay_memory.max_reward)
                logger.record_tabular("min_reward", my_replay_memory.min_reward)
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
                        if not only_pretrain:
                            logger.record_tabular("bc_loss", np.mean(bc_loss_list[-100:]))
                        else:
                            logger.record_tabular("bc_loss", 0)

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
                if terminal_life_lost and args.env_id == "BreakoutDeterministic-v4":
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
