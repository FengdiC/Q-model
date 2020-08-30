import tensorflow as tf
import numpy as np
import os
import random
import math
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')
import utils
import PriorityBuffer
import logger


class DQN:
    """Implements a Deep Q Network"""
    def __init__(self, args, n_actions=2, hidden=512, grid = 10,agent='dqfd', name="dqn"):
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
        self.args = args
        self.n_actions = n_actions
        self.hidden = hidden
        self.grid = grid

        self.input = tf.placeholder(shape=[None, self.grid*self.grid],dtype=tf.float32)
        self.weight = tf.placeholder(shape=[None,],dtype=tf.float32)
        self.policy = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.diff = tf.placeholder(shape=[None, ], dtype=tf.float32)
        #layers
        self.layer1 = tf.layers.dense(inputs= self.input, units = hidden, activation=tf.nn.relu,
                                      kernel_initializer=tf.variance_scaling_initializer(scale=2),name='fc1')
        self.layer2 = tf.layers.dense(inputs=self.layer1, units=hidden, activation=tf.nn.relu,
                                      kernel_initializer=tf.variance_scaling_initializer(scale=2), name='fc2')
        self.dense = tf.layers.dense(inputs=self.d, units=hidden,
                                   kernel_initializer=tf.variance_scaling_initializer(scale=2), name="fc3")

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
        self.action_prob = tf.nn.softmax(self.q_values)
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

        MAIN_DQN_VARS = tf.trainable_variables(scope=name)
        if agent == "dqn":
            print("DQN Loss")
            self.loss, self.loss_per_sample = self.dqn_loss(MAIN_DQN_VARS)
        elif agent == "expert":
            print("Expert Loss")
            self.loss, self.loss_per_sample = self.expert_loss(MAIN_DQN_VARS)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        self.update = self.optimizer.minimize(self.loss)

    def expert_loss(self, t_vars):
        self.prob = tf.reduce_sum(tf.multiply(self.action_prob, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)),
                               axis=1)
        self.posterior = self.target_q+self.diff *(1-self.prob)
        l_dq = tf.losses.huber_loss(labels=self.posterior, predictions=self.Q, weights=self.weight*self.policy,
                                    reduction=tf.losses.Reduction.NONE)
        l_n_dq = tf.losses.huber_loss(labels=self.target_n_q, predictions=self.Q, weights=self.weight*self.policy,
                                      reduction=tf.losses.Reduction.NONE)

        l2_reg_loss = 0
        for v in t_vars:
            if 'bias' not in v.name:
                l2_reg_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.dqfd_l2
        self.l2_reg_loss = l2_reg_loss
        self.l_dq = l_dq
        self.l_n_dq = l_n_dq
        self.l_jeq = self.diff *(1-self.prob)/(self.target_q+0.001)

        loss_per_sample = l_dq + self.args.LAMBDA_1 * l_n_dq
        loss = tf.reduce_mean(loss_per_sample+l2_reg_loss)
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

        loss_per_sample = l_dq + self.args.LAMBDA_1 * l_n_dq
        loss = tf.reduce_mean(loss_per_sample+l2_reg_loss)
        return loss, loss_per_sample


def learn(session, states, actions, diffs, rewards, new_states, terminal_flags,weights, expert_idxes, n_step_rewards, n_step_states,
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
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})
    double_q = q_vals[range(batch_size), arg_q_max]

    prob = session.run(target_dqn.action_prob, feed_dict={target_dqn.input:states})
    action_prob = prob[range(batch_size), actions]


    mask = np.where(diffs>0,np.ones((batch_size,)),np.zeros((batch_size,)))
    action_prob = mask * action_prob + (1-mask) * np.ones((batch_size,))
    action_prob = action_prob ** 0.2
    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = rewards + (gamma*double_q *  (1-terminal_flags))


    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:n_step_states})
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:n_step_states})
    double_q = q_vals[range(batch_size), arg_q_max]
    target_n_q = n_step_rewards + (gamma*double_q * not_terminal)
    loss_sample, l_dq, l_n_dq, l_jeq, _ = session.run([main_dqn.loss_per_sample, main_dqn.l_dq, main_dqn.l_n_dq,
                                                main_dqn.l_jeq, main_dqn.update],
                          feed_dict={main_dqn.input:states,
                                     main_dqn.target_q:target_q,
                                     main_dqn.action:actions,
                                     main_dqn.target_n_q:target_n_q,
                                     main_dqn.expert_state:expert_idxes,
                                     main_dqn.weight:weights,
                                     main_dqn.diff: diffs,
                                     main_dqn.policy:action_prob
                                     })
    return loss_sample, np.mean(l_dq), np.mean(l_n_dq), np.mean(l_jeq)

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
            n_step_rewards, n_step_states, last_step_gamma, not_terminal = replay_buffer.compute_n_step_target_q(idxes, args.dqfd_n_step,
                                                                                                                 args.gamma)
            loss, loss_dq, loss_dq_n, loss_jeq = learn(sess, generated_states, generated_actions, generated_diffs,generated_rewards,
                                                       generated_new_states, generated_terminal_flags, generated_weights,
                                                       expert_idxes, n_step_rewards,n_step_states, last_step_gamma, not_terminal,
                                                       MAIN_DQN, TARGET_DQN, BS,DISCOUNT_FACTOR, args)
            episode_dq_loss.append(loss_dq)
            episode_dq_n_loss.append(loss_dq_n)
            episode_jeq_loss.append(loss_jeq)

            replay_buffer.update_priorities(idxes, loss, expert_idxes, frame_num, expert_priority_decay=args.expert_priority_decay,
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

    return episode_reward_sum, episode_length, np.mean(episode_loss),np.mean(episode_jeq_loss), \
           time.time() - start_time, np.mean(expert_ratio)

def train( priority=True):
    args = utils.argsparser()
    name = args.agent
    tf.random.set_random_seed(args.seed)
    np.random.seed(args.seed)


    MAX_EPISODE_LENGTH = args.max_eps_len       # Equivalent of 5 minutes of gameplay at 60 frames per second
    EVAL_FREQUENCY = args.eval_freq          # Number of frames the agent sees between evaluations
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
    print("Agent: ", name)
    with tf.variable_scope('mainDQN'):
        MAIN_DQN = DQN(args, atari.env.action_space.n, HIDDEN,agent=name, name="mainDQN")
    with tf.variable_scope('targetDQN'):
        TARGET_DQN = DQN(args, atari.env.action_space.n, HIDDEN,agent=name, name="targetDQN")

    init = tf.global_variables_initializer()
    MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
    TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if priority:
        my_replay_memory = PriorityBuffer.PrioritizedReplayBuffer(MEMORY_SIZE, args.alpha,args.var, agent=name)
    else:
        my_replay_memory = PriorityBuffer.ReplayBuffer(MEMORY_SIZE, agent=name)
    network_updater = utils.TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
    action_getter = utils.ActionGetter(atari.env.action_space.n,
                                 replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                                 max_frames=MAX_FRAMES,
                                 eps_initial=args.initial_exploration)
    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session(config=config)
    sess.run(init)
    # saver.restore(sess, "../models/" + name + "/" + args.env_id + "/"  + "model-" + str(5614555))

    frame_number = REPLAY_MEMORY_START_SIZE
    eps_number = 0

    if not os.path.exists("./" + args.gif_dir + "/" + name + "/" + args.env_id + "/"):
        os.makedirs("./" + args.gif_dir + "/" + name + "/" + args.env_id + "/")
    if not os.path.exists("./" + args.checkpoint_dir + "/" + name + "/" + args.env_id + "/"):
        os.makedirs("./" + args.checkpoint_dir + "/" + name + "/" + args.env_id + "/")
    if not os.path.exists("./" + args.log_dir + "/" + name + "/" + args.env_id + "/"):
        os.makedirs("./" + args.log_dir + "/" + name + "/" + args.env_id + "/")

    logger.configure("./" + args.log_dir + "/" + name + "/" + args.env_id + "/")

    if os.path.exists(args.expert_dir + args.expert_file):
        my_replay_memory.load_expert_data( args.expert_dir + args.expert_file)
    else:
        print("No Expert Data ... ")
    #Pretrain step ..
    logger.log("Starting {agent} for Environment '{env}'".format(agent='dqfd', env=args.env_id))
    logger.log("Start pre-training")
    if args.pretrain_bc_iter > 0:
        utils.train_step_dqfd(sess, args, MAIN_DQN, TARGET_DQN, network_updater, action_getter, my_replay_memory, atari, 0,
                              args.pretrain_bc_iter, learn, pretrain=True)
        print("done pretraining ,test prioritized buffer")
        print("buffer expert size: ",my_replay_memory.expert_idx)
        print("expert priorities: ",my_replay_memory._it_sum.sum(my_replay_memory.agent_history_length, my_replay_memory.expert_idx))
    if name=='dqn':
        print("agent dqn!")
        my_replay_memory.delete_expert(MEMORY_SIZE)
    else:
        print("Agent: ", name)

    utils.evaluate_model(sess, args, EVAL_STEPS, MAIN_DQN, action_getter, MAX_EPISODE_LENGTH, atari, frame_number,
                         model_name=name, gif=True, random=False)
    utils.build_initial_replay_buffer(sess, atari, my_replay_memory, action_getter, MAX_EPISODE_LENGTH, REPLAY_MEMORY_START_SIZE,
                                      MAIN_DQN, args)
    episode_reward_list = []
    episode_len_list = []
    episode_loss_list = []
    episode_time_list = []
    episode_expert_list = []
    episode_jeq_list = []

    print_iter = 25
    last_eval = 0
    eval_reward = 0
    while frame_number < MAX_FRAMES:
        eps_rw, eps_len, eps_loss, eps_jeq_loss,eps_time, exp_ratio = utils.train_step_dqfd(sess, args, MAIN_DQN, TARGET_DQN, network_updater,
                                                                               action_getter, my_replay_memory, atari, frame_number,
                                                                               MAX_EPISODE_LENGTH, learn, pretrain=False)
        episode_reward_list.append(eps_rw)
        episode_len_list.append(eps_len)
        episode_loss_list.append(eps_loss)
        episode_time_list.append(eps_time)
        episode_expert_list.append(exp_ratio)
        episode_jeq_list.append(eps_jeq_loss)

        frame_number += eps_len
        eps_number += 1
        last_eval += eps_len
        if len(episode_len_list) % print_iter == 0:
            # print("Last " + str(print_iter) + " Episodes Reward: ", np.mean(episode_reward_list[-print_iter:]))
            # print("Last " + str(print_iter) + " Episodes Length: ", np.mean(episode_len_list[-print_iter:]))
            # print("Last " + str(print_iter) + " Episodes Loss: ", np.mean(episode_loss_list[-print_iter:]))
            # print("Last " + str(print_iter) + " Episodes Time: ", np.mean(episode_time_list[-print_iter:]))
            # print("Last " + str(print_iter) + " Reward/Frames: ", np.sum(episode_reward_list[-print_iter:])/np.sum(episode_len_list[-print_iter:]))
            # print("Last " + str(print_iter) + " Expert Ratio: ", np.mean(episode_expert_list[-print_iter:]))
            # print("Total " + str(print_iter) + " Episode time: ", np.sum(episode_time_list[-print_iter:]))
            # print("Current Exploration: ", action_getter.get_eps(frame_number))
            # print("Frame Number: ", frame_number)
            # print("Episode Number: ", eps_number)
            #print("size of buffer: ",my_replay_memory.count)
            if my_replay_memory.expert_idx > my_replay_memory.agent_history_length:
              print("expert priorities: ",my_replay_memory._it_sum.sum(my_replay_memory.agent_history_length, my_replay_memory.expert_idx))
            #print("total priorities: ",
            #      my_replay_memory._it_sum.sum(my_replay_memory.agent_history_length, my_replay_memory.count -1))
            logger.record_tabular("Last " + str(print_iter) + " Episodes Reward: ",
                                  np.mean(episode_reward_list[-print_iter:]))
            logger.record_tabular("Last " + str(print_iter) + " Episodes Length: ",
                                  np.mean(episode_len_list[-print_iter:]))
            logger.record_tabular("Last " + str(print_iter) + " Episodes Loss: ",
                                  np.mean(episode_loss_list[-print_iter:]))
            logger.record_tabular("Last " + str(print_iter) + " Episodes Time: ",
                                  np.mean(episode_time_list[-print_iter:]))
            logger.record_tabular("Last " + str(print_iter) + " Reward/Frames: ",
                                  np.sum(episode_reward_list[-print_iter:]) / np.sum(episode_len_list[-print_iter:]))
            logger.record_tabular("Last " + str(print_iter) + " Expert Ratio: ",
                                  np.mean(episode_expert_list[-print_iter:]))
            logger.record_tabular("Total " + str(print_iter) + " Episode time: ",
                                  np.sum(episode_time_list[-print_iter:]))
            logger.record_tabular("Current Exploration: ", action_getter.get_eps(frame_number))
            logger.record_tabular("Frame Number: ", frame_number)
            logger.record_tabular("Episode Number: ", eps_number)
            logger.record_tabular("Eval Reward: ",eval_reward)
            logger.dumpkvs()

        if EVAL_FREQUENCY < last_eval:
            last_eval = 0
            eval_reward, eval_var = utils.evaluate_model(sess, args, EVAL_STEPS, MAIN_DQN, action_getter, MAX_EPISODE_LENGTH, atari,
                                 frame_number, model_name=name, gif=True)
            logger.log("Evaluation result: ",eval_reward, ":::",eval_var)
            logger.dumpkvs()
            #saver.save(sess, "./" + args.checkpoint_dir + "/" + name + "/" + args.env_id + "/" + "model",
            #        global_step=frame_number)
train()