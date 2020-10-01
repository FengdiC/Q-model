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
import logger

from tensorboard_logger import tensorflowboard_logger

def softargmax(x, beta=1e10):
  x = tf.convert_to_tensor(x)
  return tf.nn.softmax(x*beta)

class DQN:
    """Implements a Deep Q Network"""
    def __init__(self, args, n_actions=4, hidden=1024,
               frame_height=84, frame_width=84, agent_history_length=4, agent='dqfd', name="dqn", max_reward=1,ratio_expert=1.0):
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
        self.max_reward = max_reward
        self.ratio_expert=ratio_expert
        self.args = args
        self.var = args.var
        self.eta = args.eta
        self.n_actions = n_actions
        self.hidden = hidden
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length

        self.input = tf.placeholder(shape=[None, self.frame_height,
                                         self.frame_width, self.agent_history_length],
                                  dtype=tf.float32)
        self.weight = tf.placeholder(shape=[None,],dtype=tf.float32)
        self.policy = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.n_policy = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.diff = tf.placeholder(shape=[None, ], dtype=tf.float32)
        # Normalizing the input
        self.inputscaled = (self.input - 127.5) / 127.5
        # self.inputscaled = self.input
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
        self.dense = tf.layers.dense(inputs=self.d, units=hidden,
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
        self.q_values = self.value + tf.subtract(self.advantage,
                                               tf.reduce_mean(self.advantage, axis=1, keepdims=True))
        self.action_prob = tf.nn.softmax(args.eta * self.q_values)

        self.nstep_minus_prob = tf.placeholder(shape=[None], dtype=tf.float32)

        self.best_action = tf.argmax(self.q_values, 1)

        # The next lines perform the parameter update. This will be explained in detail later.

        # targetQ according to Bellman equation:
        # Q = r + gamma*max Q', calculated in the function learn()
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        # Action that was performed
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        self.target_n_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.expert_state = tf.placeholder(shape=[None], dtype=tf.float32)

        # Q value of the action that was performed
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)),
                             axis=1)
        # Parameter updates
        #self.loss_per_sample = tf.losses.huber_loss(labels=self.target_q, predictions=self.Q,
                                                  # reduction=tf.losses.Reduction.NONE)
        #self.loss = tf.reduce_mean(self.loss_per_sample)

        MAIN_DQN_VARS = tf.trainable_variables(scope=name)
        if agent == "dqn":
            print("DQN Loss")
            self.loss, self.loss_per_sample = self.dqn_loss(MAIN_DQN_VARS)
        if agent == "dqn_with_priority_weight":
            print("DQN priority weights loss")
            self.loss, self.loss_per_sample = self.dqn_with_priority_weight_loss(MAIN_DQN_VARS)
        elif agent == "baseline_dqn":
            print("Baseline DQN")
            self.loss, self.loss_per_sample = self.baseline_dqn_loss(MAIN_DQN_VARS)
        elif agent == "expert":
            print("Expert Loss")
            self.loss, self.loss_per_sample = self.expert_loss(MAIN_DQN_VARS,decay=args.decay)

        elif agent == "expert_diff_policy":
            print("Expert Loss off policy")
            self.loss, self.loss_per_sample = self.expert_loss_diff_policy(MAIN_DQN_VARS,decay=args.decay)

        elif agent == "dqfd_with_priority_weight":
            print("DQFD with priority")
            self.loss, self.loss_per_sample = self.dqfd_loss_with_priority_weights(MAIN_DQN_VARS)
        elif agent == "dqfd_all_weights":
            print("DQFD all weights")
            self.loss, self.loss_per_sample = self.loss_dqfd_all_weights(MAIN_DQN_VARS)
        else:
            # print("DQFD")
            # self.loss, self.loss_per_sample = self.dqfd_loss(MAIN_DQN_VARS)
            raise NotImplementedError
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        self.update = self.optimizer.minimize(self.loss)

    def loss_jeq(self):
        expert_act_one_hot = tf.one_hot(self.action, self.n_actions, dtype=tf.float32)
        self.expert_q_value = tf.reduce_sum(expert_act_one_hot * self.q_values, axis=1)
        #JEQ = max_{a in A}(Q(s,a) + l(a_e, a)) - Q(s, a_e)
        self.q_plus_margin = self.q_values + (1-expert_act_one_hot)*self.args.dqfd_margin
        self.max_q_plus_margin = tf.reduce_max(self.q_plus_margin, axis=1)
        jeq = (self.max_q_plus_margin - self.expert_q_value) * self.expert_state
        return jeq

    def loss_dqfd_all_weights(self, t_vars):
        l_dq = tf.losses.huber_loss(labels=self.target_q, predictions=self.Q, weights=self.weight*self.policy,
                                    reduction=tf.losses.Reduction.NONE)
        l_n_dq = tf.losses.huber_loss(labels=self.target_n_q, predictions=self.Q, weights=self.weight*self.policy,
                                      reduction=tf.losses.Reduction.NONE)
        l_jeq = self.loss_jeq()
        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq
        self.l_n_dq = l_n_dq
        self.l_jeq = l_jeq

        loss_per_sample = l_dq + self.args.LAMBDA_1 * l_n_dq
        loss = tf.reduce_mean(loss_per_sample+l2_reg_loss + self.args.LAMBDA_2 * l_jeq)
        return loss, loss_per_sample

    def dqfd_loss_with_priority_weights(self, t_vars):
        l_dq = tf.losses.huber_loss(labels=self.target_q, predictions=self.Q, weights=self.weight,
                                    reduction=tf.losses.Reduction.NONE)
        l_n_dq = tf.losses.huber_loss(labels=self.target_n_q, predictions=self.Q, weights=self.weight,
                                      reduction=tf.losses.Reduction.NONE)
        l_jeq = self.loss_jeq()
        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq
        self.l_n_dq = l_n_dq
        self.l_jeq = l_jeq

        loss_per_sample = self.l_dq + self.args.LAMBDA_1 * self.l_n_dq
        loss = tf.reduce_mean(loss_per_sample+l2_reg_loss + self.args.LAMBDA_2 * self.l_jeq)
        return loss, loss_per_sample

    def dqfd_loss(self, t_vars):
        l_dq = tf.losses.huber_loss(labels=self.target_q, predictions=self.Q,
                                    reduction=tf.losses.Reduction.NONE)
        l_n_dq = tf.losses.huber_loss(labels=self.target_n_q, predictions=self.Q,
                                      reduction=tf.losses.Reduction.NONE)
        l_jeq = self.loss_jeq()

        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq
        self.l_n_dq = l_n_dq
        self.l_jeq = l_jeq

        loss_per_sample = self.l_dq + self.args.LAMBDA_1 * self.l_n_dq
        loss = tf.reduce_mean(loss_per_sample+self.l2_reg_loss+ self.args.LAMBDA_2 * self.l_jeq)
        return loss, loss_per_sample


    def expert_loss_diff_policy(self, t_vars,decay='t', loss_cap=None):
        # decay 't' means order one decay 1/(beta+t)
        #       's' means beta^2+t/(beta+t)^2
        # here set beta = 4
        if decay =='t':
          ratio = 1/(4+self.diff)
        elif decay =='s':
          ratio = (16+4*self.diff)/tf.square(4+self.diff)
        self.prob = tf.reduce_sum(tf.multiply(self.action_prob, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)),
                               axis=1)
        
        self.posterior = self.target_q+self.eta*self.var*ratio *(1-self.prob)*self.expert_state
        l_dq = tf.losses.huber_loss(labels=self.posterior, predictions=self.Q, weights=self.weight*self.policy,
                                    reduction=tf.losses.Reduction.NONE)

        self.n_posterior = self.target_n_q + 0.4*self.eta*self.var*ratio * self.nstep_minus_prob * self.expert_state
        l_n_dq = tf.losses.huber_loss(labels=self.n_posterior, predictions=self.Q, weights=self.weight*self.n_policy,
                                      reduction=tf.losses.Reduction.NONE)

        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq
        self.l_n_dq = l_n_dq
        self.l_jeq = 0.4*self.eta*self.var*ratio *(1-self.prob)/(self.target_q+0.001)

        loss_per_sample = self.l_dq + self.args.LAMBDA_1 * self.l_n_dq
        loss = tf.reduce_mean(loss_per_sample+self.l2_reg_loss)
        return loss, loss_per_sample

    def expert_loss(self, t_vars, decay='t', loss_cap=None):
        # decay 't' means order one decay 1/(beta+t)
        #       's' means beta^2+t/(beta+t)^2
        # here set beta = 3
        if decay =='t':
          ratio = 1/(4+self.diff)
        elif decay =='s':
          ratio = (16+4*self.diff)/tf.square(4+self.diff)
        self.prob = tf.reduce_sum(tf.multiply(self.action_prob, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)),
                               axis=1)
        
        self.posterior = self.target_q+self.eta*self.var*ratio *(1-self.prob)*self.expert_state
        l_dq = tf.losses.huber_loss(labels=self.posterior, predictions=self.Q, weights=self.weight*self.policy,
                                    reduction=tf.losses.Reduction.NONE)

        self.n_posterior = self.target_n_q +self.eta*self.var*ratio * self.nstep_minus_prob * self.expert_state
        l_n_dq = tf.losses.huber_loss(labels=self.n_posterior, predictions=self.Q, weights=self.weight*self.n_policy,
                                      reduction=tf.losses.Reduction.NONE)

        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq
        self.l_n_dq = l_n_dq
        self.l_jeq = self.eta*self.var*ratio *(1-self.prob)/(self.target_q+0.001)

        loss_per_sample = self.l_dq + self.args.LAMBDA_1 * self.l_n_dq
        loss = tf.reduce_mean(loss_per_sample+self.l2_reg_loss)
        return loss, loss_per_sample


    def dqn_loss(self,t_vars):
        l_dq = tf.losses.huber_loss(labels=self.target_q, predictions=self.Q, reduction=tf.losses.Reduction.NONE)
        l_n_dq = tf.losses.huber_loss(labels=self.target_n_q, predictions=self.Q, reduction=tf.losses.Reduction.NONE)
        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq
        self.l_n_dq = l_n_dq
        self.l_jeq = tf.constant(0)

        loss_per_sample = self.l_dq + self.args.LAMBDA_1 * self.l_n_dq
        loss = tf.reduce_mean(loss_per_sample+self.l2_reg_loss)
        return loss, loss_per_sample

    def dqn_with_priority_weight_loss(self,t_vars):
        l_dq = tf.losses.huber_loss(labels=self.target_q, predictions=self.Q, weights=self.weight, reduction=tf.losses.Reduction.NONE)
        l_n_dq = tf.losses.huber_loss(labels=self.target_n_q, predictions=self.Q, weights=self.weight, reduction=tf.losses.Reduction.NONE)
        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq
        self.l_n_dq = l_n_dq
        self.l_jeq = tf.constant(0)

        loss_per_sample = self.l_dq + self.args.LAMBDA_1 * self.l_n_dq
        loss = tf.reduce_mean(loss_per_sample+self.l2_reg_loss)
        return loss, loss_per_sample


    def baseline_dqn_loss(self,t_vars):
        l_dq = tf.losses.huber_loss(labels=self.target_q, predictions=self.Q, reduction=tf.losses.Reduction.NONE)
        #l_n_dq = tf.losses.huber_loss(labels=self.target_n_q, predictions=self.Q, reduction=tf.losses.Reduction.NONE)
        # l2_reg_loss = 0
        # for v in t_vars:
        #     if 'bias' not in v.name:
        #         l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        # self.l2_reg_loss = l2_reg_loss
        self.l2_reg_loss = tf.constant(0)
        self.l_dq = l_dq
        self.l_n_dq = tf.constant(0)
        self.l_jeq = tf.constant(0)
        loss_per_sample = self.l_dq
        loss = tf.reduce_mean(loss_per_sample)
        return loss, loss_per_sample

def learn(session, states, actions, diffs, rewards, new_states, terminal_flags,weights, expert_idxes, n_step_rewards, n_step_states, n_step_actions,
          last_step_gamma, not_terminal, main_dqn, target_dqn, batch_size, gamma, args):
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

    prob = session.run(target_dqn.action_prob, feed_dict={target_dqn.input:states})
    basic_action_prob = prob[range(batch_size), actions]

    mask = np.where(diffs>0,np.ones((batch_size,)),np.zeros((batch_size,)))
    action_prob = mask * basic_action_prob + (1-mask) * np.ones((batch_size,))
    action_prob = action_prob ** args.power


    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = rewards + (gamma*double_q *  (1-terminal_flags))
    #target_pos = np.where(target_q>=0,np.zeros(batch_size),np.ones(batch_size))
    #if np.sum(target_pos)>0:
    #  print("Check if target values are positive: ", np.sum(target_pos))

    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:n_step_states[:, -1]})
    # The target network estimates the Q-values (in the next state s', new_states is passed!)
    # for every transition in the minibatch
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:n_step_states[:, -1]})
    double_q = q_vals[range(batch_size), arg_q_max]
    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_n_q = n_step_rewards[:, -1] + (gamma*double_q * not_terminal[:, -1])
    # Gradient descend step to update the parameters of the main network
    #print(np.max(rewards), np.min(rewards))

    # self.l2_reg_loss = l2_reg_loss
    # self.l_dq = l_dq
    # self.l_n_dq = l_n_dq
    # self.l_jeq = l_jeq

    n_step_prob = np.zeros((batch_size, ))
    n_policy = np.ones(batch_size)
    if np.sum(expert_idxes)>0:
      idx = np.nonzero(expert_idxes)[0]
      # print("expert input size: ",n_step_states[idx].shape)
      n_state_list = []
      n_action_list = []
      n_expert_idxes = []
      for i in range(batch_size):
          for j in range(n_step_states.shape[1]):
            n_state_list.append(np.expand_dims(n_step_states[i, j], axis=0))
            n_action_list.append(n_step_actions[i, j])
            n_expert_idxes.append(idx[i])
      n_step_states = np.concatenate(n_state_list, axis=0)
      n_step_actions = np.array(n_action_list)
      n_expert_idxes = np.array(n_expert_idxes)

      all_prob = session.run(target_dqn.prob,
                         feed_dict={target_dqn.input: n_step_states[n_expert_idxes], target_dqn.action: n_step_actions[n_expert_idxes]})

      n_policy_expert = []
      n_step_prob_expert = []
      for i in range(all_prob.shape[0]//args.dqfd_n_step):
          #should be a dataset of expert by n_actions
          prob = all_prob[i * args.dqfd_n_step:(i + 1) * args.dqfd_n_step]
          n_policy_expert.append(prob)
          n_step_prob_expert.append(np.sum((1 - prob) * gamma ** i))
      n_step_prob[idx] = np.array(n_step_prob_expert)
      n_policy_expert = (np.prod(np.array(n_policy_expert), axis=1)) ** 0.2
      n_policy[idx] = n_policy_expert


      # nstep_minus_prob = []
      # nstep_prob = []
      # BS = n_step_actions[idx].shape[0]
      # # print("check batch size: ",BS)
      # for i in range(n_step_rewards.shape[1]):
      #     prob = all_prob[i * BS:(i + 1) * BS]
      #     nstep_prob.append(prob)
      #     nstep_minus_prob.append((1 - prob)*gamma**i)
      #
      # nstep_prob = np.array(nstep_prob)[0:1]
      # maxx = np.ndarray.max(np.prod(np.array(nstep_prob),axis=0))
      # nstep_policy = (np.prod(np.array(nstep_prob),axis=0))**0.2
      # print(maxx,":::",nstep_policy[0],":::",action_prob[0])
      # quit()
      # nstep_minus_prob = np.sum(np.array(nstep_minus_prob), axis=0)
      #
      # n_step_prob[idx] = nstep_minus_prob
      # n_policy[idx] = nstep_policy


      
      # if np.sum(expert_idxes)>1 and np.sum(expert_idxes)<30:
      #  print("check n step oof policy ratio: ",nstep_policy)
    loss_sample, l_dq, l_n_dq, l_jeq, l_l2,_ = session.run([main_dqn.loss_per_sample, main_dqn.l_dq, main_dqn.l_n_dq,
                                                main_dqn.l_jeq, main_dqn.l2_reg_loss, main_dqn.update],
                          feed_dict={main_dqn.input:states,
                                     main_dqn.target_q:target_q,
                                     main_dqn.action:actions,
                                     main_dqn.target_n_q:target_n_q,
                                     main_dqn.expert_state:expert_idxes,
                                     main_dqn.weight:weights,
                                     main_dqn.diff: diffs,
                                     main_dqn.policy:action_prob,
                                     main_dqn.n_policy:n_policy,
                                     main_dqn.nstep_minus_prob: n_step_prob
                                     })
    # print(loss, q_val.shape, q_values.shape)
    # for i in range(batch_size):
    #     if loss[i] > 5:
    #         print(i, loss[i], q_val[i], target_q[i], target_n_q[i], q_values[i], actions[i], expert_idxes[i])
    # if np.sum(terminal_flags) > 0:
    #     quit()
    return loss_sample, np.mean(l_dq), np.mean(l_n_dq), np.mean(l_jeq), np.mean(l_l2), np.mean(mask)

def train( priority=True):
    args = utils.argsparser()
    name = args.agent
    tf.random.set_random_seed(args.seed)
    np.random.seed(args.seed)


    MAX_EPISODE_LENGTH = args.max_eps_len       # Equivalent of 5 minutes of gameplay at 60 frames per second
    EVAL_FREQUENCY = args.eval_freq          # Number of frames the agent sees between evaluations
    GIF_FREQUENCY = args.gif_freq
    EVAL_STEPS = args.eval_len               # Number of frames for one evaluation
    var = args.var                           # initial variance value

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
    print("Agent: ", name)
    if priority:
        print("Priority")
        my_replay_memory = PriorityBuffer.PrioritizedReplayBuffer(MEMORY_SIZE, args.alpha, agent=name, batch_size=args.batch_size)
    else:
        print("Not Priority")
        my_replay_memory = PriorityBuffer.ReplayBuffer(MEMORY_SIZE, agent=name, batch_size=args.batch_size)

    # saver.restore(sess, "../models/" + name + "/" + args.env_id + "/"  + "model-" + str(5614555))
    frame_number = REPLAY_MEMORY_START_SIZE
    eps_number = 0

    if not os.path.exists("./" + args.gif_dir + "/" + name + "/" + args.env_id + "_seed_" + str(args.seed) + "/"):
        os.makedirs("./" + args.gif_dir + "/" + name + "/" + args.env_id + "_seed_" + str(args.seed) + "/")
    if not os.path.exists("./" + args.checkpoint_dir + "/" + name + "/" + args.env_id + "_seed_" + str(args.seed) + "/"):
        os.makedirs("./" + args.checkpoint_dir + "/" + name + "/" + args.env_id + "_seed_" + str(args.seed) + "/")
    print("Expert Directory: ", args.expert_dir + args.expert_file)
    if os.path.exists(args.expert_dir + args.expert_file):
        max_reward, num_expert = my_replay_memory.load_expert_data( args.expert_dir + args.expert_file)
    else:
        print("No Expert Data ... ")
        
    ratio_expert = float(num_expert/(MEMORY_SIZE-num_expert))**args.power
    #ratio_expert=0.01
    with tf.variable_scope('mainDQN'):
        MAIN_DQN = DQN(args, atari.env.action_space.n, HIDDEN,agent=name, name="mainDQN", max_reward=max_reward,ratio_expert=ratio_expert)
    with tf.variable_scope('targetDQN'):
        TARGET_DQN = DQN(args, atari.env.action_space.n, HIDDEN,agent=name, name="targetDQN", max_reward=max_reward,ratio_expert=ratio_expert)

    init = tf.global_variables_initializer()
    MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
    TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    network_updater = utils.TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
    action_getter = utils.ActionGetter(atari.env.action_space.n,
                                 replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                 max_frames=MAX_FRAMES,
                                 eps_initial=args.initial_exploration)
    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session(config=config)
    sess.run(init)

    tflogger = tensorflowboard_logger("./" + args.log_dir + "/" + name + "_" + args.env_id + "_priority_" + str(priority) + "_seed_" + str(args.seed) + "_" + args.custom_id, sess, args)

    #Pretrain step ..
    if args.pretrain_bc_iter > 0:
        utils.train_step_dqfd(sess, args, MAIN_DQN, TARGET_DQN, network_updater, action_getter, my_replay_memory, atari, 0,
                              args.pretrain_bc_iter, learn, pretrain=True)
        print("done pretraining ,test prioritized buffer")
        print("buffer expert size: ",my_replay_memory.expert_idx)
        tflogger.log_scalar("Expert Priorities", my_replay_memory._it_sum.sum(my_replay_memory.agent_history_length, my_replay_memory.expert_idx), frame_number)

    if args.delete_expert:
        print("Expert data deleted .... ")
        my_replay_memory.delete_expert(MEMORY_SIZE)
    print("Agent: ", name)

    eval_reward, eval_var = utils.evaluate_model(sess, args, EVAL_STEPS, MAIN_DQN, action_getter, MAX_EPISODE_LENGTH, atari, frame_number,
                         model_name=name, gif=False, random=False)
    tflogger.log_scalar("Evaluation/Reward", eval_reward, frame_number)
    tflogger.log_scalar("Evaluation/Reward Variance", eval_var, frame_number)
    utils.build_initial_replay_buffer(sess, atari, my_replay_memory, action_getter, MAX_EPISODE_LENGTH, REPLAY_MEMORY_START_SIZE,
                                      MAIN_DQN, args)
    print_iter = 25
    last_eval = 0
    last_gif = 0
    initial_time = time.time()
    max_eval_reward = -1

    while frame_number < MAX_FRAMES:
        eps_rw, eps_len, eps_loss, eps_dq_loss, eps_dq_n_loss, eps_jeq_loss, eps_l2_loss,eps_time, exp_ratio, gen_weight_mean, \
        gen_weight_std, non_expert_gen_diff, expert_gen_diff, mean_mask = utils.train_step_dqfd(sess, args, MAIN_DQN, TARGET_DQN,
                                                                          network_updater,action_getter, my_replay_memory, atari,
                                                                          frame_number,MAX_EPISODE_LENGTH, learn, pretrain=False)
        frame_number += eps_len
        eps_number += 1
        last_gif += 1
        last_eval += eps_len

        if eps_number % print_iter == 0:
            if my_replay_memory.expert_idx > my_replay_memory.agent_history_length:
                tflogger.log_scalar("Expert Priorities", my_replay_memory._it_sum.sum(my_replay_memory.agent_history_length, my_replay_memory.expert_idx), frame_number)
                tflogger.log_scalar("Total Priorities", my_replay_memory._it_sum.sum(my_replay_memory.agent_history_length, my_replay_memory.count -1), frame_number)

        tflogger.log_scalar("Episode/Reward", eps_rw, frame_number)
        tflogger.log_scalar("Episode/Length", eps_len, frame_number)
        tflogger.log_scalar("Episode/Loss/Total", eps_loss, frame_number)
        tflogger.log_scalar("Episode/Time", eps_time, frame_number)
        tflogger.log_scalar("Episode/Reward Per Frame", eps_rw / max(1, eps_len),
                            frame_number)
        tflogger.log_scalar("Episode/Expert Ratio", exp_ratio, frame_number)
        tflogger.log_scalar("Episode/Exploration", action_getter.get_eps(frame_number), frame_number)
        tflogger.log_scalar("Episode/Loss/DQ", eps_dq_loss, frame_number)
        tflogger.log_scalar("Episode/Loss/DQ N-step", eps_dq_n_loss, frame_number)
        tflogger.log_scalar("Episode/Loss/JEQ", eps_jeq_loss, frame_number)
        tflogger.log_scalar("Episode/Loss/L2", eps_l2_loss, frame_number)
        tflogger.log_scalar("Episode/Weight/Mean", gen_weight_mean, frame_number)
        tflogger.log_scalar("Episode/Weight/Std", gen_weight_std, frame_number)
        tflogger.log_scalar("Episode/Time", eps_time, frame_number)
        tflogger.log_scalar("Episode/Reward/Reward Per Frame", eps_rw / max(1, eps_len),
                            frame_number)
        tflogger.log_scalar("Episode/Expert Ratio", exp_ratio, frame_number)
        tflogger.log_scalar("Episode/Exploration", action_getter.get_eps(frame_number), frame_number)
        tflogger.log_scalar("Episode/Non Expert Gen Diff", non_expert_gen_diff, frame_number)
        tflogger.log_scalar("Episode/Expert Gen Diff", expert_gen_diff, frame_number)
        #tflogger.log_scalar("Episode/Mean Mask", mean_mask, frame_number)

        tflogger.log_scalar("Total Episodes", eps_number, frame_number)
        tflogger.log_scalar("Replay Buffer Size", my_replay_memory.count, frame_number)
        tflogger.log_scalar("Elapsed Time", time.time() - initial_time, frame_number)
        tflogger.log_scalar("Frames Per Hour", frame_number/((time.time() - initial_time)/3600), frame_number)

        if EVAL_FREQUENCY <= last_eval:
            last_eval = last_eval - EVAL_FREQUENCY
            if GIF_FREQUENCY <= last_gif:
                last_gif = last_gif - GIF_FREQUENCY
                eval_reward, eval_var = utils.evaluate_model(sess, args, EVAL_STEPS, MAIN_DQN, action_getter, MAX_EPISODE_LENGTH, atari,
                                 frame_number, model_name=name, gif=True)
            else:
                eval_reward, eval_var = utils.evaluate_model(sess, args, EVAL_STEPS, MAIN_DQN, action_getter, MAX_EPISODE_LENGTH, atari,
                                 frame_number, model_name=name, gif=False)
            if eval_reward > max_eval_reward:
                max_eval_reward = eval_reward

            tflogger.log_scalar("Evaluation/Max_Reward", max_eval_reward, frame_number)
            tflogger.log_scalar("Evaluation/Reward", eval_reward, frame_number)
            tflogger.log_scalar("Evaluation/Reward Variance", eval_var, frame_number)
            saver.save(sess, "./" + args.checkpoint_dir + "/" + name + "/" + args.env_id +  "_seed_" + str(args.seed) + "/" + "model",
                   global_step=frame_number)
train()
