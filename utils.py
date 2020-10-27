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
    parser.add_argument('--agent', help='the trainer used', type=str, default='dqfd')
    parser.add_argument('--expert_dir', type=str, default='./')
    parser.add_argument('--expert_file', type=str, default='human_SeaquestDeterministic-v4_0.pkl')
    parser.add_argument('--expert_file_path', type=str, default='None')

    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='models')
    parser.add_argument('--checkpoint_index', type=int, help='index of model to load', default=-1)
    parser.add_argument('--checkpoint_file_path', type=str, default='None')
    parser.add_argument('--special_tag', type=str, default='')

    parser.add_argument('--log_dir', help='the directory to save log file', default='logs')
    parser.add_argument('--gif_dir', help='the directory to save GIFs file', default='gifs')
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    parser.add_argument('--num_sampled', type=int, help='Num Generated Sequence', default=1)
    parser.add_argument('--max_eps_len', type=int, help='Max Episode Length', default=18000)
    parser.add_argument('--gif_freq', type=int, help='Gif Frequency', default=500000)
    parser.add_argument('--eval_freq', type=int, help='Evaluation Frequency', default=50000)
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

    parser.add_argument('--alpha', type=float, help='Max Episode Length', default=0.6)
    parser.add_argument('--beta', type=float, help='Max Episode Length', default=0.4)
    parser.add_argument('--var', type=float, help='Variance of prior Q-values', default=2.3*9.5618) #test 1, 3.5, 5.5, test large values
    parser.add_argument('--eta', type=float, help='Action prob coefficient', default=3.0)
    parser.add_argument('--decay', type=str, help='Decay Computation of Variance of prior Q-values', default='s')
    parser.add_argument('--power', type=float, help='Off policy correction power', default=0.2)

    parser.add_argument('--decay_rate', type=int, help='Max Episode Length', default=1000000)
    parser.add_argument('--max_ent_coef_bc', type=float, help='Max Episode Length', default=1.0)
    parser.add_argument('--pretrain_bc_iter', type=int, help='Max Episode Length', default=5000)

    parser.add_argument('--LAMBDA_1', type=float, help='Lambda 1 for expert', default=1)
    parser.add_argument('--LAMBDA_2', type=float, help='Lambda 1 for expert', default=1)

    parser.add_argument('--expert_priority_modifier', type=int, help='Max Episode Length', default=4)
    parser.add_argument('--min_expert_priority', type=int, help='Max Episode Length', default=0.05)

    parser.add_argument('--dqfd_l2', type=int, help='Lambda 1 for expert', default=0.00001)
    parser.add_argument('--dqfd_margin', type=float, help='Lambda 1 for expert', default=0.8)
    parser.add_argument('--dqfd_n_step', type=int, help='Lambda 1 for expert', default=10)
    parser.add_argument('--delete_expert', type=int, help='0 for false', default=0)
    parser.add_argument('--grid_size', type=int, help='for toy examples', default=128)


    parser.add_argument('--env_id', type=str, default='SeaquestDeterministic-v4')
    parser.add_argument('--stochastic_exploration', type=str, default="False")

    parser.add_argument('--load_frame_num', type=int, help='If load model 0, else load frame num ....', default=0)
    parser.add_argument('--initial_exploration', type=float, help='Amount of exploration at start', default=1.0)
    parser.add_argument('--stochastic_environment', type=str, choices=['True', 'False'], default='False')
    parser.add_argument('--custom_id', type=str, default='')
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
            # 
            if args.stochastic_exploration == "True":
                action = action_getter.get_stochastic_action(sess, atari.state, MAIN_DQN)
            else:
                action = action_getter.get_action(sess, 0, atari.state, MAIN_DQN)
            # print("Action: ",action)
            # 
            next_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)
            #  Store transition in the replay memory
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



def evaluate_model(sess, args, eval_steps, MAIN_DQN, action_getter, max_eps_len, atari, 
                    frame_num, model_name="dqn", window=5, gif=False, random=False):
    frames_for_gif = []
    eval_rewards = []
    gif_q_values = []
    gif_terminal_values = []
    gif_reward_values = []
    evaluate_frame_number = 0
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
                    if args.stochastic_exploration == "True":
                        action = action_getter.get_stochastic_action(sess, frame_num,
                                                        atari.state,
                                                        x_values,
                                                        MAIN_DQN,
                                                        evaluation=True)
                    else:
                        action = action_getter.get_action(sess, frame_num,
                                                        atari.state,
                                                        MAIN_DQN,
                                                        evaluation=True)
                else:
                    action = action_getter.get_random_action()
            processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
            evaluate_frame_number += 1
            episode_reward_sum += reward
            if gif:
                #get the q_value ... 
                q_values = sess.run(MAIN_DQN.q_values, feed_dict={MAIN_DQN.input:[atari.state]})
                gif_q_values.append(q_values)
                frames_for_gif.append(new_frame)
                gif_terminal_values.append(terminal_life_lost)
                gif_reward_values.append(reward)
            if terminal and len(eval_rewards) == 0:
                gif = False  # Save only the first game of the evaluation as a gif
                break
            elif terminal:
                break
        if terminal:
            eval_rewards.append(episode_reward_sum)
        print("Eval", len(eval_rewards), "Eval Reward", episode_reward_sum, "frame_num", evaluate_frame_number)
        # if len(eval_rewards) % 10 == 0:
        #     print("Evaluation Completion: ", str(evaluate_frame_number) + "/" + str(eval_steps))

    if len(eval_rewards) == 0:
        eval_rewards.append(episode_reward_sum)
    if len(frames_for_gif) > 0:
        if not os.path.exists("./" + args.gif_dir + "/" + model_name + "/" + args.env_id  + "_seed_" + str(args.seed) +  "/gif_figures/"):
            os.makedirs("./" + args.gif_dir + "/" + model_name + "/" + args.env_id  + "_seed_" + str(args.seed) +  "/gif_figures/")

        if len(frames_for_gif) == max_eps_len:
            gif_terminal_values[-1] = 1
        diff_list = []
        td_error_list = []
        max_q_value_list = []

        for i in range(len(frames_for_gif) - 1):
            if gif_terminal_values[i]:
                diff_list.append(0)
                max_q_value_list.append(0)
                td_error_list.append(0)
                continue
            diff_list.append(np.sum(np.abs(np.max(gif_q_values[i + 1]) - np.max(gif_q_values[i]))))
            adjusted_rewards = np.sign(gif_reward_values[i]) * np.log(1+np.abs(gif_reward_values[i]))
            max_q_value_list.append(np.max(gif_q_values[i]))            
            target_q_value = adjusted_rewards + (1 - gif_terminal_values[i]) * args.gamma * np.max(gif_q_values[i + 1])
            td_error = huber(1, np.max(gif_q_values[i]) - target_q_value)
            td_error_list.append(td_error)

        def reject_outliers(data, m = 10):
            d = np.abs(data - np.median(data))
            mdev = np.median(d)
            s = d/mdev if mdev else 0.
            return data[s<m]

        diff_array = np.array(diff_list)
        td_array = np.array(td_error_list)
        cleaned_data = reject_outliers(diff_array)
        std = np.std(cleaned_data)
        mean = np.median(cleaned_data)

        cleaned_data = reject_outliers(td_array)
        std_td = np.std(cleaned_data)
        mean_td = np.mean(cleaned_data)

        #diff_list[np.abs(diff_list - mean) > 6 * std] = 6 * std #cap the outliers ... 
        max_td_error = np.max(td_error_list)
        min_td_error = np.min(td_error_list)
        # scaled_td = (np.array(td_error_list) - np.mean(td_error_list))/np.maximum(0.01, np.std(td_error_list))
        # scaled_q_value = (np.array(max_q_value_list) - np.mean(max_q_value_list))/np.maximum(0.01, np.std(max_q_value_list))
        # scaled_diff = (np.array(diff_list) - np.mean(diff_list))/np.maximum(0.01, np.std(diff_list))
        q_value_array = np.array(max_q_value_list)

        outlier_list = np.ones((len(frames_for_gif,)))
        lower_outlier_list = np.ones((len(frames_for_gif,)))
        outlier_sign = np.ones((len(frames_for_gif,)))

        # if np.sum(gif_terminal_values) > 1:
        #     terminal_indices = np.nonzero(gif_terminal_values)[0]
        #     print(terminal_indices)

        terminal_indices = np.nonzero(gif_terminal_values)[0]
        if not hasattr(MAIN_DQN, 'q_prediction_ability'):
            MAIN_DQN.q_prediction_ability = []
            MAIN_DQN.td_prediction_ability = [] 
        if len(terminal_indices) > 1:
            #find a prediction length 
            q_diff_term_pred = []
            td_term_pred = []
            for idx in terminal_indices:
                q_diff_term_pred.append(0)
                td_term_pred.append(0)
                for i in range(max(0, idx - 30), idx):
                    if np.abs(diff_array[i] - mean) > 3 * std:
                        q_diff_term_pred[-1] = max(q_diff_term_pred[-1], idx - i)
                    if np.abs(td_array[i] - mean_td) > 3 * std_td:
                        td_term_pred[-1] = max(td_term_pred[-1], idx - i)
            
            MAIN_DQN.q_prediction_ability.append(np.mean(q_diff_term_pred))
            MAIN_DQN.td_prediction_ability.append(np.mean(td_term_pred))
            print("Prediction: ", q_diff_term_pred, td_term_pred)
        else:
            MAIN_DQN.q_prediction_ability.append(0)
            MAIN_DQN.td_prediction_ability.append(0)

        plt.plot(np.arange(len(MAIN_DQN.q_prediction_ability)), MAIN_DQN.q_prediction_ability, label="Q Pred Turns");
        plt.plot(np.arange(len(MAIN_DQN.td_prediction_ability)), MAIN_DQN.td_prediction_ability, label="TD Pred Turns");
        plt.title("Q-value and TD Death Predictions(Steps before death")
        plt.legend(bbox_to_anchor=(0.75, 0.95), loc='upper left')
        plt.savefig("./" + args.gif_dir + "/" + model_name + "/" + args.env_id  + "_seed_" + str(args.seed) +  "/gif_figures/death_pred.png")
        plt.close()

        for i in range(len(frames_for_gif) - 1):
            if gif_terminal_values[i]:
                continue
            if gif_terminal_values[i + 1]:
                continue
            if np.abs(diff_array[i] - mean) > 3 * std:
                lower_bound = max(0, i - window)
                higher_bound = min(i + window, len(frames_for_gif))
                if np.sum(gif_terminal_values[lower_bound:i]) > 0:
                    #there is a terminal .... 
                    for terminal_index in range(lower_bound, i):
                        if gif_terminal_values[terminal_index]:
                            break
                    lower_bound = terminal_index + 1
                if np.sum(gif_terminal_values[i:higher_bound]) > 0:
                    #there is a terminal .... 
                    for terminal_index in range(i, higher_bound):
                        if gif_terminal_values[terminal_index]:
                            break
                    higher_bound = min(i+1, terminal_index - 1)
                outlier_list[lower_bound:higher_bound] = 0
                lower_outlier_list[i:higher_bound] = 0
                outlier_sign[lower_bound:higher_bound] = np.sign(np.sum(np.max(gif_q_values[i + 1]) - np.max(gif_q_values[i])))

        low_gifs = []
        normal_gifs = []
        plt.plot(np.arange(td_array.shape[0]), td_array, label="TD Loss");
        plt.title("TD Loss(Huber) vs Timestep(t)")
        plt.savefig("./" + args.gif_dir + "/" + model_name + "/" + args.env_id  + "_seed_" + str(args.seed) +  "/gif_figures/" + str(frame_num) + "_td_loss.png")
        plt.close()

        plt.plot(np.arange(q_value_array.shape[0]), q_value_array, label="Max Q");
        plt.title("max Q(s,a) vs Timestep(t)")
        #plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.savefig("./" + args.gif_dir + "/" + model_name + "/" + args.env_id  + "_seed_" + str(args.seed) +  "/gif_figures/" + str(frame_num) + "_max_q_value.png")
        plt.close()

        plt.plot(np.arange(diff_array.shape[0]), diff_array, label="max Q(s', a') - Q(s, a)");
        plt.title("max Q(s', a') - Q(s, a) vs Timestep(t)")
        #plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.savefig("./" + args.gif_dir + "/" + model_name + "/" + args.env_id  + "_seed_" + str(args.seed) +  "/gif_figures/" + str(frame_num) + "_q_diff.png")
        plt.close()

        plt.bar(np.arange(outlier_list.shape[0]), (1 - outlier_list) * outlier_sign, label="Q_diff > 4", width=2);
        plt.title("Q_diff > 3 std vs Timestep(t)")
        #plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.savefig("./" + args.gif_dir + "/" + model_name + "/" + args.env_id  + "_seed_" + str(args.seed) +  "/gif_figures/" + str(frame_num) + "_outlier.png")
        plt.close()

        max_reward = np.max(gif_reward_values)
        plt.bar(np.arange(len(gif_reward_values)), np.array(gif_reward_values), label="Reward", width=10)
        plt.bar(np.arange(len(gif_terminal_values)), -1 * max(1, np.max(gif_reward_values)) * np.array(gif_terminal_values), label="Termination", width=10)
        plt.title("Reward + Termination vs Timestep(t)")
        plt.savefig("./" + args.gif_dir + "/" + model_name + "/" + args.env_id  + "_seed_" + str(args.seed) +  "/gif_figures/" + str(frame_num) + "_reward_termination.png")
        plt.close()

        for i in range(len(frames_for_gif)):    
            if lower_outlier_list[i] == 0:
                #positive, green
                #negative, blue
                mean_frame = np.expand_dims(np.mean(frames_for_gif[i], axis=2), axis=2)
                if outlier_sign[i] == 1:
                    gif_frame = np.concatenate([mean_frame, np.expand_dims(np.copy(frames_for_gif[i][:,:,1]), axis=2), mean_frame], axis=2)
                else:
                    gif_frame = np.concatenate([mean_frame, mean_frame, np.expand_dims(np.copy(frames_for_gif[i][:,:,2]), axis=2)], axis=2)
                #slowmo for focused states ... 
                for _ in range(12):
                    low_gifs.append(np.copy(gif_frame))
            else:
                low_gifs.append(frames_for_gif[i])

            # print("Q_values:", gif_q_values[i])
            # if gif_terminal_values[i]:
            #     print("Frame:", i, "Out:", outlier_list[i], "\n\n")
            # else:
            #     print("Frame:", i, "Reward:", gif_reward_values[i], "TD Loss", td_error_list[i], "Diff: ", np.sum(np.abs(np.max(gif_q_values[i + 1]) - np.max(gif_q_values[i]))), outlier_sign[i], "Out:", outlier_list[i])
            if outlier_list[i] == 0:
                mean_frame = np.expand_dims(np.mean(frames_for_gif[i], axis=2), axis=2)
                if outlier_sign[i] == 1:
                    gif_frame = np.concatenate([mean_frame, np.expand_dims(np.copy(frames_for_gif[i][:,:,1]), axis=2), mean_frame], axis=2)
                else:
                    gif_frame = np.concatenate([mean_frame, mean_frame, np.expand_dims(np.copy(frames_for_gif[i][:,:,2]), axis=2)], axis=2)
                for _ in range(6):
                    normal_gifs.append(np.copy(gif_frame))
            else:
                normal_gifs.append(frames_for_gif[i])
        print(np.sum(gif_terminal_values))
        print("Mean Difference: ", mean, "STD: ", std, "Ratio:", 1 - np.mean(outlier_list))
        try:
            generate_gif(frame_num, frames_for_gif, eval_rewards[0],
                            "./" + args.gif_dir + "/" + model_name + "/" + args.env_id  + "_seed_" + str(args.seed) +  "/" + "gif_")
            generate_gif(frame_num, low_gifs, eval_rewards[0],
                            "./" + args.gif_dir + "/" + model_name + "/" + args.env_id  + "_seed_" + str(args.seed) +  "/" + "low_bound_gif_")
        except IndexError:
            print("No evaluation game finished")
    return np.mean(eval_rewards),np.var(eval_rewards)

# def evaluate_model(sess, args, eval_steps, MAIN_DQN, action_getter, max_eps_len, atari, frame_num, model_name="dqn", gif=False, random=False):
#     frames_for_gif = []
#     eval_rewards = []
#     evaluate_frame_number = 0
#     total_reward = 0
#     if random:
#         print("Random Action Evaluation Baseline .... ")
#     else:
#         print("Evaluating Current Models .... ")
#     while evaluate_frame_number < eval_steps:
#         terminal_life_lost, _ = atari.reset(sess, evaluation=True)
#         episode_reward_sum = 0
#         for _ in range(max_eps_len):
#             # Fire (action 1), when a life was lost or the game just started,
#             # so that the agent does not stand around doing nothing. When playing
#             # with other environments, you might want to change this...
#             if terminal_life_lost and args.env_id == "BreakoutDeterministic-v4":
#                 action = 1
#             else:
#                 if not random:
#                     action = action_getter.get_action(sess, frame_num,
#                                                       atari.state,
#                                                       MAIN_DQN,
#                                                       evaluation=True)
#                 else:
#                     action = action_getter.get_random_action()

#             processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
#             evaluate_frame_number += 1
#             episode_reward_sum += reward
#             total_reward += reward
#             if gif:
#                 frames_for_gif.append(new_frame)
#             if terminal and len(eval_rewards) == 0:
#                 eval_rewards.append(episode_reward_sum)
#                 gif = False  # Save only the first game of the evaluation as a gif
#                 break
#         if len(eval_rewards) % 10 == 0:
#             print("Evaluation Completion: ", str(evaluate_frame_number) + "/" + str(eval_steps))
#     eval_rewards.append(episode_reward_sum)
#     # print("\n\n\n-------------------------------------------")
#     # print("Evaluation score:\n", np.mean(eval_rewards),':::',np.var(eval_rewards))
#     # print("-------------------------------------------\n\n\n")
#     try:
#         if len(frames_for_gif) > 0:
#             generate_gif(frame_num, frames_for_gif, eval_rewards[0],
#                         "./" + args.gif_dir + "/" + model_name + "/" + args.env_id  + "_seed_" + str(args.seed) +  "/" + "gif_")
#     except IndexError:
#         print("No evaluation game finished")
#     return np.mean(eval_rewards),np.var(eval_rewards)


def train_step_dqfd(sess, args, MAIN_DQN, TARGET_DQN, network_updater, action_getter, replay_buffer, atari, frame_num, eps_length,
                    learn, pretrain=False):
    start_time = time.time()
    terminal_life_lost, _ = atari.reset(sess, evaluation=True)
    episode_reward_sum = 0
    episode_length = 0
    episode_loss = []

    episode_dq_loss = []
    episode_dq_n_loss = []
    episode_jeq_loss = []
    episode_l2_loss = []

    expert_ratio = []
    episode_weights = []
    episode_diff_non_expert = []
    episode_diff_expert = []
    episode_mask_mean = []

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
            next_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)
            frame_num += 1
            episode_reward_sum += reward
            episode_length += 1

            replay_buffer.add(obs_t=next_frame[:, :, 0], reward=reward, action=action, done=terminal_life_lost)
        else:
            frame_num += 1
            if frame_num % 1000 == 0:
                print("Current Loss: ", frame_num, np.mean(episode_loss[-1000:]), np.mean(episode_dq_loss[-1000:]),
                      np.mean(episode_dq_n_loss[-1000:]), np.mean(episode_jeq_loss[-1000:]))

        if frame_num % UPDATE_FREQ == 0 or pretrain:
            if pretrain:
                generated_states, generated_actions, generated_diffs, generated_rewards, generated_new_states, \
                generated_terminal_flags, generated_weights, idxes, expert_idxes = replay_buffer.sample(
                    BS, args.beta, expert=pretrain, random=True)  # Generated trajectories
            else:
                generated_states, generated_actions, generated_diffs, generated_rewards, generated_new_states, \
                generated_terminal_flags, generated_weights, idxes, expert_idxes = replay_buffer.sample(
                    BS, args.beta, expert=pretrain)  # Generated trajectories
            episode_weights.append(generated_weights)
            # n_step_rewards, n_step_states, last_step_gamma, not_terminal = replay_buffer.compute_n_step_target_q(idxes, args.dqfd_n_step,
            #                                                                                                      args.gamma)
            n_step_rewards, n_step_states, n_step_actions, last_step_gamma, not_terminal = replay_buffer.compute_all_n_step_target_q(idxes,
                                                                                                                 args.dqfd_n_step,
                                                                                                                 args.gamma)

            loss, loss_dq, loss_dq_n, loss_jeq, loss_l2, mask = learn(sess, generated_states, generated_actions, generated_diffs,generated_rewards,
                                                       generated_new_states, generated_terminal_flags, generated_weights,
                                                       expert_idxes, n_step_rewards,n_step_states, n_step_actions, last_step_gamma, not_terminal,
                                                       MAIN_DQN, TARGET_DQN, BS,DISCOUNT_FACTOR, args)
            episode_mask_mean.append(np.mean(mask))
            episode_dq_loss.append(loss_dq)
            episode_dq_n_loss.append(loss_dq_n)
            episode_jeq_loss.append(loss_jeq)
            episode_l2_loss.append(loss_l2)
            episode_diff_non_expert.append(np.sum(generated_diffs * (1 - expert_idxes)) /max(1, np.sum(1 - expert_idxes)))
            episode_diff_expert.append(np.sum(generated_diffs * expert_idxes) /max(1, np.sum(expert_idxes)))
            replay_buffer.update_priorities(idxes, loss, expert_idxes, frame_num, expert_priority_modifier=args.expert_priority_modifier,
                                            min_expert_priority=args.min_expert_priority,pretrain = pretrain)
            expert_ratio.append(np.sum(expert_idxes)/BS)
            episode_loss.append(loss)
        if pretrain:
            if frame_num % (NETW_UPDATE_FREQ//UPDATE_FREQ) == 0 and frame_num > 0:
                print("UPDATING Network ... ")
                network_updater.update_networks(sess)  
        else:
            if frame_num % NETW_UPDATE_FREQ == 0 and frame_num > 0:
                print("UPDATING Network ... ")
                network_updater.update_networks(sess)  
            if terminal:
                break
    episode_weights = np.concatenate(episode_weights, axis=0)
    return episode_reward_sum, episode_length, np.mean(episode_loss),np.mean(episode_dq_loss), np.mean(episode_dq_n_loss), np.mean(episode_jeq_loss), np.mean(episode_l2_loss), \
           time.time() - start_time, np.mean(expert_ratio), np.mean(episode_weights), np.std(episode_weights), np.mean(episode_diff_non_expert), np.mean(episode_diff_expert), np.mean(episode_mask_mean)

def train_step(sess, args, MAIN_DQN, TARGET_DQN, network_updater, action_getter, replay_buffer, atari, frame_num, eps_length, learn,
               pretrain=False, priority=False):
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

            # Store transition in the replay memory
            replay_buffer.add(obs_t=next_frame[:, :, 0], reward=reward, action=action, done=terminal_life_lost)

        if frame_num % UPDATE_FREQ == 0 or pretrain:
            if not priority:
                generated_states, generated_actions,generated_diffs, generated_rewards, generated_new_states, generated_terminal_flags = replay_buffer.sample(BS, expert=pretrain)  # Generated trajectories
                loss = learn(sess, generated_states, generated_actions, generated_diffs,generated_rewards, generated_new_states,
                             generated_terminal_flags, MAIN_DQN, TARGET_DQN, BS, DISCOUNT_FACTOR, args)  # (8�?
                episode_loss.append(loss)
                expert_ratio.append(1)
            else:
                if pretrain:
                    generated_states, generated_actions, generated_diffs,generated_rewards, generated_new_states, generated_terminal_flags, _, idxes, expert_idxes = replay_buffer.sample(BS, args.beta, expert=pretrain, random=True)  # Generated trajectories
                else:
                    generated_states, generated_actions, generated_diffs,generated_rewards, generated_new_states, generated_terminal_flags, _, idxes, expert_idxes = replay_buffer.sample(BS, args.beta, expert=pretrain)  # Generated trajectories

                loss = learn(sess, generated_states, generated_actions, generated_diffs, generated_rewards, generated_new_states,
                             generated_terminal_flags, MAIN_DQN, TARGET_DQN, BS, DISCOUNT_FACTOR, args)  # (8�?
                replay_buffer.update_priorities(idxes, loss, expert_idxes, frame_num,
                                                expert_priority_modifier=args.expert_priority_modifier,
                                                min_expert_priority=args.min_expert_priority, pretrain=pretrain)
                expert_ratio.append(np.sum(expert_idxes)/BS)
                episode_loss.append(loss)

        if pretrain:
            if frame_num % (NETW_UPDATE_FREQ//UPDATE_FREQ) == 0 and frame_num > 0:
                print("UPDATING Network ... ")
                network_updater.update_networks(sess)  # 
        else:
            if frame_num % NETW_UPDATE_FREQ == 0 and frame_num > 0:
                print("UPDATING Network ... ")
                network_updater.update_networks(sess)  # 
            if terminal:
                break

    return episode_reward_sum, episode_length, np.mean(episode_loss), time.time() - start_time, np.mean(expert_ratio)
