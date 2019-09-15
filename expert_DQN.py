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
        self.expert_data_loss = tf.reduce_mean(-self.expert_prob * tf.log(self.expert_prob  +0.00001) * self.expert_weights)# + a l2 reg to prevent overtraining too much
        self.generated_input = tf.placeholder(shape=[None, self.frame_height,
                                           self.frame_width, self.agent_history_length],
                                    dtype=tf.float32)


        _, self.generated_action_preference = self.build_graph(self.generated_input, hidden, n_actions, reuse=True)
        self.action_prob_generated = tf.nn.softmax(self.generated_action_preference)
        self.generated_data_loss = tf.reduce_mean(self.action_prob_generated * tf.log(self.action_prob_generated  +0.00001))# + a l2 reg to prevent overtraining too much
        self.expert_loss = self.expert_data_loss + self.generated_data_loss
        self.expert_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.expert_update =self.expert_optimizer.minimize(self.expert_loss, var_list=expert_vars)

        self.action_prob = self.action_prob_q * self.action_prob_expert
        self.best_action = tf.argmax(self.action_prob, 1)
        #
        # self.all_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # self.total_update =self.all_optimizer.minimize(self.expert_loss * 0.05 + self.loss, var_list=t_vars)
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
            self.conv4 = tf.layers.conv2d(
                inputs=self.conv3, filters=hidden, kernel_size=[7, 7], strides=1,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4', reuse=reuse)
            self.d = tf.layers.flatten(self.conv4)
            self.dense = tf.layers.dense(inputs = self.d,units = hidden,activation=tf.tanh,
                                         kernel_initializer=tf.variance_scaling_initializer(scale=2), name="fc5", reuse=reuse)
            q_values = tf.layers.dense(
                inputs=self.dense, units=n_actions,activation=tf.identity,
                kernel_initializer=tf.variance_scaling_initializer(scale=2), name='value', reuse=reuse)

        with tf.variable_scope("ExpertPref"):
        #     self.expert_conv1 = tf.layers.conv2d(
        #         inputs=self.scaled_input, filters=32, kernel_size=[8, 8], strides=4,
        #         kernel_initializer=tf.variance_scaling_initializer(scale=2),
        #         padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1', reuse=reuse)
        #     self.expert_conv2 = tf.layers.conv2d(
        #         inputs=self.expert_conv1, filters=64, kernel_size=[4, 4], strides=2,
        #         kernel_initializer=tf.variance_scaling_initializer(scale=2),
        #         padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2', reuse=reuse)
        #     self.expert_conv3 = tf.layers.conv2d(
        #         inputs=self.expert_conv2, filters=64, kernel_size=[3, 3], strides=1,
        #         kernel_initializer=tf.variance_scaling_initializer(scale=2),
        #         padding="valid", activation=tf.nn.relu, use_bias=False, name='conv3', reuse=reuse)
        #     self.expert_conv4 = tf.layers.conv2d(
        #         inputs=self.expert_conv3, filters=hidden, kernel_size=[7, 7], strides=1,
        #         kernel_initializer=tf.variance_scaling_initializer(scale=2),
        #         padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4', reuse=reuse)
        #     self.expert_d = tf.layers.flatten(self.expert_conv4)
            self.expert_dense = tf.layers.dense(inputs =self.d,units = hidden,activation=tf.tanh,
                                         kernel_initializer=tf.variance_scaling_initializer(scale=2), name="fc5", reuse=reuse)
            action_preference = tf.layers.dense(
                inputs=self.expert_dense, units=n_actions,activation=tf.identity,
                kernel_initializer=tf.variance_scaling_initializer(scale=2), name='action_preference', reuse=reuse)
        return q_values, action_preference

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
    dataset.batch_size = batch_size
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

def test_q_values(sess, env, action_getter, input, output, threshold_test=0.9):
    #Test number of states with greater than 90% confidence
    my_replay_memory = utils.ReplayMemory(size=MEMORY_SIZE, batch_size=BS)  # (★)
    env.reset(sess)
    accumulated_rewards = 0
    terminal_life_lost = True
    for i in range(20000):
        action = 1 if terminal_life_lost else action_getter.get_action(sess, 1000,
                                                                        atari.state,
                                                                        MAIN_DQN,
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
    env.reset(sess)
    count = 0
    for i in range(my_replay_memory.count):
        states, actions, rewards, new_states, terminal_flags = my_replay_memory.get_minibatch()
        values = sess.run(output, feed_dict={input: states})
        if np.max(values) >= threshold_test:
            count += 1
    print("Number of states: ", my_replay_memory.count)
    print("Number of certain actions: ", count)


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
    parser.add_argument('--lr', type=float, help='Max Episode Length', default=0.0000625)
    parser.add_argument('--stochastic_exploration', type=str, default="False")
    parser.add_argument('--initial_exploration', type=float, help='Amount of exploration at start', default=1.0)
    parser.add_argument('--env_id', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--stochastic', type=str, choices=['True', 'False'], default='True')
    return parser.parse_args()

args = argsparser()
tf.random.set_random_seed(args.seed)
np.random.seed(args.seed)
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
atari = utils.Atari(args.env_id, args.stochastic, NO_OP_STEPS)
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
    dataset = pickle.load(open(args.expert_dir +  args.env_id + "/" + "seed_" + str(args.seed) + "/" + args.expert_file, "rb"))
    utils.generate_weights(dataset)

    my_replay_memory = utils.ReplayMemory(size=MEMORY_SIZE, batch_size=BS)  # (★)
    network_updater = utils.TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
    action_getter = utils.ActionGetter(atari.env.action_space.n,
                                 replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                 max_frames=MAX_FRAMES,
                                 eps_initial=args.initial_exploration)

    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session(config=config)
    sess.run(init)
    fixed_state = np.expand_dims(atari.fixed_state(sess),axis=0)

    if args.checkpoint_index >= 0:
        saver.restore(sess, args.checkpoint_dir +  args.env_id + "/" + "seed_" + str(args.seed) + "/" + "model--" + str(args.checkpoint_index))
        print("Loaded Model ... ", args.checkpoint_dir +  args.env_id + "seed_" + str(args.seed) + "/" + "model--" + str(args.checkpoint_index))
    logger.configure(args.log_dir +  args.env_id + "/" + "seed_" + str(args.seed) + "/")
    if not os.path.exists(args.gif_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/"):
        os.makedirs(args.gif_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/")
    if not os.path.exists(args.checkpoint_dir + args.env_id + "/"+ "seed_" + str(args.seed) + "/"):
        os.makedirs(args.checkpoint_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/")

    frame_number = 0
    rewards = []
    loss_list = []
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
                    loss,expert_loss = learn(sess, dataset,my_replay_memory, MAIN_DQN, TARGET_DQN,
                                 BS, DISCOUNT_FACTOR)  # (8★)
                    loss_list.append(loss)
                    expert_loss_list.append(expert_loss)
                if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                    network_updater.update_networks(sess)  # (9★)

                if terminal:
                    terminal = False
                    break
            rewards.append(episode_reward_sum)
            episode_length_list.append(episode_length)
            
            # Output the progress:
            if len(rewards) % 10 == 0:
                # logger.log("Runing frame number {0}".format(frame_number))
                test_q_values(sess, atari, action_getter, MAIN_DQN.input, MAIN_DQN.action_prob_expert)
                logger.record_tabular("frame_number",frame_number)
                logger.record_tabular("training_reward",np.mean(rewards[-100:]))
                logger.record_tabular("expert update loss", np.mean(expert_loss_list))
                logger.record_tabular("episode length",np.mean(episode_length_list[-100:]))
                logger.record_tabular("Current Exploration", action_getter.get_eps(frame_number))
                q_vals = sess.run(MAIN_DQN.q_values, feed_dict={MAIN_DQN.input: fixed_state})
                for i in range(atari.env.action_space.n):
                    logger.record_tabular("q_val action {0}".format(i),q_vals[0,i])

                q_vals = sess.run(MAIN_DQN.action_prob_q, feed_dict={MAIN_DQN.input: fixed_state})
                for i in range(atari.env.action_space.n):
                    print("Q Prob: ", i, q_vals[0,i])
                q_vals = sess.run(MAIN_DQN.action_prob_expert, feed_dict={MAIN_DQN.input: fixed_state})
                for i in range(atari.env.action_space.n):
                    print("Expert Prob: ", i, q_vals[0,i])
                q_vals = sess.run(MAIN_DQN.action_prob, feed_dict={MAIN_DQN.input: fixed_state})
                for i in range(atari.env.action_space.n):
                    print("Total Prob: ", i, q_vals[0,i])

                print("Completion: ", str(epoch_frame)+"/"+str(EVAL_FREQUENCY))
                print("Current Frame: ",frame_number)
                print("Average Reward: ", np.mean(rewards[-100:]))
                print("TD Loss: ", np.mean(loss_list[-100:]))
                if frame_number > REPLAY_MEMORY_START_SIZE:
                    print("Average Expert Loss: ", np.mean(expert_loss_list[-100:]))

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
            utils.generate_gif(frame_number, frames_for_gif, eval_rewards[0], args.gif_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/")
        except IndexError:
            print("No evaluation game finished")
        logger.log("Average Evaluation Reward", np.mean(eval_rewards))
        logger.log("Average Sequence Length", evaluate_frame_number/max(1, len(eval_rewards)))
        # Save the network parameters
        saver.save(sess, args.checkpoint_dir + args.env_id + "/" + "seed_" + str(args.seed) + "/" + 'model-', global_step=frame_number)
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
