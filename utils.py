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
    parser.add_argument('--eval_len', type=int, help='Max Episode Length', default=18000)
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

    parser.add_argument('--alpha', type=float, help='Max Episode Length', default=0.4)
    parser.add_argument('--beta', type=float, help='Max Episode Length', default=0.6)
    parser.add_argument('--expert_weight', type=float, help='Max Episode Length', default=1.0)

    parser.add_argument('--decay_rate', type=int, help='Max Episode Length', default=1000000)
    parser.add_argument('--max_ent_coef_bc', type=float, help='Max Episode Length', default=1.0)
    parser.add_argument('--pretrain_bc_iter', type=int, help='Max Episode Length', default=10000)


    parser.add_argument('--LAMBDA_1', type=float, help='Lambda 1 for expert', default=0.1)
    parser.add_argument('--LAMBDA_2', type=float, help='Lambda 1 for expert', default=1)
    parser.add_argument('--dqfd_l2', type=int, help='Lambda 1 for expert', default=0.00001)
    parser.add_argument('--dqfd_margin', type=float, help='Lambda 1 for expert', default=0.8)
    parser.add_argument('--dqfd_n_step', type=int, help='Lambda 1 for expert', default=10)


    parser.add_argument('--env_id', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--stochastic_exploration', type=str, default="False")
    parser.add_argument('--initial_exploration', type=float, help='Amount of exploration at start', default=1.0)
    parser.add_argument('--stochastic_environment', type=str, choices=['True', 'False'], default='False')
    return parser.parse_args()


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
        #state = (state - 127.5)/127.5
        if evaluation:
            return session.run(main_dqn.best_action, feed_dict={main_dqn.input: [state]})[0]
        else:
            action_prob = session.run(main_dqn.action_prob, feed_dict={main_dqn.input: [state]})[0]
            rand = np.random.uniform(0, 1)
            accumulated_val = 0
            index = -1
            for i in range(action_prob.shape[0]):
                accumulated_val += action_prob[i]
                if rand < accumulated_val:
                    index = i
                    break
            #print(index, rand, action_prob)
        return index

    def get_random_action(self):
        return np.random.randint(0, self.n_actions)

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
        #state = (state - 127.5)/127.5
        if evaluation:
            eps = self.eps_evaluation
        elif frame_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif frame_number >= self.replay_memory_start_size and frame_number < self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope_2 * frame_number + self.intercept_2
        if np.random.uniform(0, 1) < eps:
            return self.get_random_action()

        #print("Was here .... ")
        result, q_vals = session.run([main_dqn.best_action, main_dqn.q_values], feed_dict={main_dqn.input: [state]})
        # result = result[0]
        # print(result, q_vals)
        return result

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
        processed_frame = self.frame_processor.process(sess, frame)  # (★★�?
        self.state = np.repeat(processed_frame, self.agent_history_length, axis=2)
        return terminal_life_lost, processed_frame

    def fixed_state(self,sess):
        frame = self.env.reset()
        for _ in range(random.randint(1, self.no_op_steps)):
            frame, _, _, _ = self.env.step(1)  # Action 'Fire'
        processed_frame = self.frame_processor.process(sess, frame)  # (★★�?
        state = np.repeat(processed_frame, self.agent_history_length, axis=2)
        return state

    def step(self, sess, action):
        """
        Args:
            sess: A Tensorflow session object
            action: Integer, action the agent performs
        Performs an action and observes the reward and terminal state from the environment
        """
        new_frame, reward, terminal, info = self.env.step(action)  # (5�?

        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']

        processed_new_frame = self.frame_processor.process(sess, new_frame)  # (6�?
        new_state = np.append(self.state[:, :, 1:], processed_new_frame, axis=2)  # (6�?
        self.state = new_state

        return processed_new_frame, reward, terminal, terminal_life_lost, new_frame


def build_initial_replay_buffer(sess, atari, my_replay_memory, action_getter, max_eps, replay_buf_size, MAIN_DQN, args):
    frame_num = 0
    while frame_num < replay_buf_size:
        _, _ = atari.reset(sess)
        for _ in range(max_eps):
            # (4�?
            if args.stochastic_exploration == "True":
                action = action_getter.get_stochastic_action(sess, atari.state, MAIN_DQN)
            else:
                action = action_getter.get_action(sess, 0, atari.state, MAIN_DQN)
            # print("Action: ",action)
            # (5�?
            next_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)
            # (7�? Store transition in the replay memory
            my_replay_memory.add(obs_t=next_frame[:, :, 0], reward=reward, action=action, done=terminal_life_lost)
            frame_num += 1
            if frame_num % (replay_buf_size//10) == 0 and frame_num > 0:
                print(frame_num)
            if terminal:
                break

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


def evaluate_model(sess, args, eval_steps, MAIN_DQN, action_getter, max_eps_len, atari, frame_num, model_name="dqn", gif=False, random=False):
    frames_for_gif = []
    eval_rewards = []
    evaluate_frame_number = 0
    total_reward = 0
    if random:
        print("Random Action Evaluation Baseline .... ")
    else:
        print("Evaluating Current Models .... ")
    while evaluate_frame_number < eval_steps:
        terminal_life_lost, _ = atari.reset(sess, evaluation=True)
        episode_reward_sum = 0
        for _ in range(max_eps_len):
            # Fire (action 1), when a life was lost or the game just started,
            # so that the agent does not stand around doing nothing. When playing
            # with other environments, you might want to change this...
            if terminal_life_lost and args.env_id == "BreakoutDeterministic-v4":
                action = 1
            else:
                if not random:
                    action = action_getter.get_action(sess, frame_num,
                                                      atari.state,
                                                      MAIN_DQN,
                                                      evaluation=True)
                else:
                    action = action_getter.get_random_action()

            processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
            evaluate_frame_number += 1
            episode_reward_sum += reward
            total_reward += reward
            if gif:
                frames_for_gif.append(new_frame)
            if terminal and len(eval_rewards) == 0:
                eval_rewards.append(episode_reward_sum)
                gif = False  # Save only the first game of the evaluation as a gif
                break
        if len(eval_rewards) % 10 == 0:
            print("Evaluation Completion: ", str(evaluate_frame_number) + "/" + str(eval_steps))
    eval_rewards.append(episode_reward_sum)
    print("\n\n\n-------------------------------------------")
    print("Evaluation score:\n", np.mean(eval_rewards))
    print("-------------------------------------------\n\n\n")
    try:
        generate_gif(frame_num, frames_for_gif, eval_rewards[0],
                     "../" + args.gif_dir + "/" + model_name + "/" + args.env_id + "/" + "gif_")
    except IndexError:
        print("No evaluation game finished")

def train_step_dqfd(sess, args, MAIN_DQN, TARGET_DQN, network_updater, action_getter, replay_buffer, atari, frame_num, eps_length, learn, pretrain=False):
    start_time = time.time()
    terminal_life_lost, _ = atari.reset(sess, evaluation=True)
    episode_reward_sum = 0
    episode_length = 0
    episode_loss = []
    expert_ratio = []

    NETW_UPDATE_FREQ = args.target_update_freq         # Number of chosen actions between updating the target network.
    DISCOUNT_FACTOR = args.gamma           # gamma in the Bellman equation
    UPDATE_FREQ = args.update_freq                  # Every four actions a gradient descend step is performed
    BS = args.batch_size
    terminal = False
    for j in range(eps_length):
        # (4�?
        if not pretrain:
            if args.stochastic_exploration == "True":
                action = action_getter.get_stochastic_action(sess, atari.state, MAIN_DQN)
            else:
                action = action_getter.get_action(sess, frame_num, atari.state, MAIN_DQN)
            # print("Action: ",action)
            # (5�?
            next_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)
            frame_num += 1
            episode_reward_sum += reward
            episode_length += 1

            # (7�? Store transition in the replay memory
            replay_buffer.add(obs_t=next_frame[:, :, 0], reward=reward, action=action, done=terminal_life_lost)
            #current_frame = next_frame
        if frame_num % UPDATE_FREQ == 0 or pretrain:
            if pretrain and j % 1000 is 0 and j > 0:
                print("Pretraining ... ", j, "Loss: ", np.mean(episode_loss[-1000:]))
            if pretrain:
                states, actions, rewards, new_states, terminal_flags, _, idxes, expert_idxes = replay_buffer.sample(BS, args.beta, expert=pretrain, random=True)  # Generated trajectories
            else:
                states, actions, rewards, new_states, terminal_flags, _, idxes, expert_idxes = replay_buffer.sample(BS, args.beta, expert=pretrain)  # Generated trajectories
            n_step_rewards, n_step_states, n_step_actions, last_step_gamma, not_terminal = replay_buffer.compute_n_step_target_q(idxes, args.dqfd_n_step, args.gamma)

            loss = learn(sess, states, actions, rewards, new_states, terminal_flags, expert_idxes, n_step_rewards, n_step_states, n_step_actions, last_step_gamma, not_terminal, MAIN_DQN, TARGET_DQN, BS, DISCOUNT_FACTOR, args)  # (8�?
            replay_buffer.update_priorities(idxes, loss, expert_idxes, expert_weight=args.expert_weight)
            expert_ratio.append(np.sum(expert_idxes) / BS)
            episode_loss.append(loss)

        if frame_num % NETW_UPDATE_FREQ == 0 and frame_num > 0:
            print("UPDATING Network ... ")
            network_updater.update_networks(sess)  # (9�?
        if terminal:
            break
    if pretrain:
        print("Loss: ", np.mean(episode_loss[-1000:]))
    return episode_reward_sum, episode_length, np.mean(episode_loss), time.time() - start_time, np.mean(expert_ratio)

def train_step(sess, args, MAIN_DQN, TARGET_DQN, network_updater, action_getter, replay_buffer, atari, frame_num, eps_length, learn, pretrain=False, priority=False):
    start_time = time.time()
    terminal_life_lost, _ = atari.reset(sess, evaluation=True)
    episode_reward_sum = 0
    episode_length = 0
    episode_loss = []
    expert_ratio = []

    NETW_UPDATE_FREQ = args.target_update_freq         # Number of chosen actions between updating the target network.
    DISCOUNT_FACTOR = args.gamma           # gamma in the Bellman equation
    UPDATE_FREQ = args.update_freq                  # Every four actions a gradient descend step is performed
    BS = args.batch_size
    terminal = False

    for _ in range(eps_length):
        if not pretrain:
            if args.stochastic_exploration == "True":
                action = action_getter.get_stochastic_action(sess, atari.state, MAIN_DQN)
            else:
                action = action_getter.get_action(sess, frame_num, atari.state, MAIN_DQN)
            # print("Action: ",action)
            # (5�?
            next_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)
            frame_num += 1
            episode_reward_sum += reward
            episode_length += 1

            # (7�? Store transition in the replay memory
            replay_buffer.add(obs_t=next_frame[:, :, 0], reward=reward, action=action, done=terminal_life_lost)

        if frame_num % UPDATE_FREQ == 0 or pretrain:
            if not priority:
                generated_states, generated_actions, generated_rewards, generated_new_states, generated_terminal_flags = replay_buffer.sample(BS, expert=pretrain)  # Generated trajectories
                loss = learn(sess, generated_states, generated_actions, generated_rewards, generated_new_states, generated_terminal_flags, MAIN_DQN, TARGET_DQN, BS, DISCOUNT_FACTOR, args)  # (8�?
                episode_loss.append(loss)
                expert_ratio.append(1)
            else:
                generated_states, generated_actions, generated_rewards, generated_new_states, generated_terminal_flags, _, idxes, expert_idxes = replay_buffer.sample(BS, args.beta, expert=pretrain)  # Generated trajectories
                loss = learn(sess, generated_states, generated_actions, generated_rewards, generated_new_states, generated_terminal_flags, MAIN_DQN, TARGET_DQN, BS, DISCOUNT_FACTOR, args)  # (8�?
                replay_buffer.update_priorities(idxes, loss, expert_idxes, expert_weight=args.expert_weight)
                expert_ratio.append(np.sum(expert_idxes)/BS)
                episode_loss.append(loss)

        if frame_num % NETW_UPDATE_FREQ == 0 and frame_num > 0:
            print("UPDATING Network ... ")
            network_updater.update_networks(sess)  # (9�?
        if terminal:
            break
    return episode_reward_sum, episode_length, np.mean(episode_loss), time.time() - start_time, np.mean(expert_ratio)
