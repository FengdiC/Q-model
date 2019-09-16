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

class expert_trajectory_dataset:
    def __init__(self, max_trajectories=20):
        print("TBD")

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
    try:
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

        plt.plot(data[convert_dict["frame_number"]], data[convert_dict["td loss"]])
        plt.title("Frame num vs tdloss")
        plt.xlabel("Frame number")
        plt.ylabel("Tdloss");
        plt.savefig(image_path + 'td_loss.png')
        plt.close()

        plt.plot(data[convert_dict["frame_number"]], data[convert_dict["Episode Length"]])
        plt.title("Frame num vs Episode Length")
        plt.xlabel("Frame number")
        plt.ylabel("Episode Length");
        plt.savefig(image_path + 'ep_len_loss.png')
        plt.close()
    except:
        print("Figure generation FAILED?")




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

    def __init__(self, envName, stochastic, no_op_steps=10, agent_history_length=4):
        if stochastic == "True":
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

def sample(args, DQN, save=True):
    tf.random.set_random_seed(args.seed)
    tf.reset_default_graph()
    np.random.seed(args.seed)
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
    atari = Atari(args.env_id, args.stochastic, NO_OP_STEPS)
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
    if args.checkpoint_index >= 0:
        saver.restore(sess, args.checkpoint_dir + args.load_model_dir + "model--" + str(args.checkpoint_index))
        print("Loaded Model ... ", args.checkpoint_dir + args.load_model_dir + "model--" + str(args.checkpoint_index))
    else:
        print("Model not found ...", args.checkpoint_dir + args.load_model_dir + "model--" + str(args.checkpoint_index))
    logger.configure(args.log_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/" + "num_traj_" + str(args.num_sampled) + "/")
    if not os.path.exists(args.gif_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/" + "num_traj_" + str(args.num_sampled) + "/"):
        os.makedirs(args.gif_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/" + "num_traj_" + str(args.num_sampled) + "/")
    if not os.path.exists(args.checkpoint_dir + args.env_id + "/seed_" + str(args.seed) + "/num_traj_" + str(args.num_sampled) + "/"):
        os.makedirs(args.checkpoint_dir + args.env_id + "/seed_" + str(args.seed) + "/num_traj_" + str(args.num_sampled) + "/")
    if not os.path.exists(args.expert_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/" + "num_traj_" + str(args.num_sampled) + "/"):
        os.makedirs(args.expert_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/" + "num_traj_" + str(args.num_sampled) + "/")
    gif = 3
    reward_list = []
    frames = 0
    log_data(args.log_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/" + "num_traj_" + str(args.num_sampled) + "/",
                   args.log_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/"+ "num_traj_" + str(args.num_sampled) + "/")
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
                        generate_gif(-gif, frames_for_gif, reward_list[-1], args.gif_dir + args.env_id + "/seed_" + str(args.seed) + "/num_traj_" + str(args.num_sampled) + "/")
                    except IndexError:
                        print(len(frames_for_gif))
                        print("No evaluation game finished")
                    gif -= 1
                reward_list.append(episode_reward_sum)
                print(len(frames_for_gif))
                break
    print("Average Reward: ", np.mean(reward_list))
    if save:
        pickle.dump(my_replay_memory, open(args.expert_dir + args.env_id + "/seed_" + str(args.seed) + "/num_traj_" + str(args.num_sampled) + "/" + args.expert_file + "_" + str(args.num_sampled), "wb"))
