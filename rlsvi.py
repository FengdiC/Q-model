import tensorflow as tf
import numpy as np
import os
import time
import sys

sys.path.append('/usr/local/lib/python3.6/dist-packages')
import utils
import PriorityBuffer
import pickle
from tensorboard_logger import tensorflowboard_logger
import math

def find_vars(name):
    t_vars = tf.trainable_variables()
    f_vars = [var for var in t_vars if name in var.name]
    return f_vars


def compute_regret(Q_value, grid , gamma, V, final_reward=1):
    pi = np.zeros((grid, grid), dtype=np.uint8)
    for i in range(grid):
        pi[i, i] = 1
    # pi = np.argmax(Q_value,axis=2)
    P = np.zeros((grid * grid, grid * grid))
    R = np.zeros((grid,grid))
    for i in range(grid-1):
        for j in range(grid-1):
            action = int(pi[i,j])
            l = i+1
            if action == 0:
                r = max(0, j - 1)
                R[i,j] = 0
            else:
                r = min(j + 1, grid - 1)
                if r == grid - 1:
                    R[i,j] = final_reward
                else:
                    R[i,j] = -0.01
            P[i*grid+j,l*grid+r]=1
    for j in range(grid-1):
        P[(grid-1)*grid+j,(grid-1)*grid+j]=1
    R = np.ndarray.flatten(R)
    Q = np.matmul(np.linalg.inv(np.eye(grid*grid)-gamma*P),R)
    return V - Q[0]

class DQN:
    """Implements a Deep Q Network"""

    def __init__(self, args, batch_size=8, n_actions=2,  grid=10, agent='dqfd', name="dqn"):
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
        self.grid = grid

        self.Q_values = tf.Variable(tf.random.truncated_normal(shape=(batch_size, grid,grid,2), mean=0, stddev=0.02), trainable=True, dtype=np.float32)
        self.one_hot_Q = tf.placeholder(shape=(batch_size, grid,grid,2),dtype=tf.float32)
        self.one_hot_q_values = tf.placeholder(shape=(batch_size, grid,grid,2),dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(self.Q_values * self.one_hot_Q,axis=3),axis=2),axis=1)
        
        self.q_values = self.Q_values * self.one_hot_q_values
        self.q_values = tf.reduce_sum(tf.reduce_sum(self.q_values, axis=2), axis=1)
        self.best_action = tf.argmax(self.q_values,1)

        self.target = tf.placeholder(shape=[batch_size], dtype=tf.float32)
        self.prior = tf.placeholder(shape=[batch_size], dtype=tf.float32)

        self.dq_loss = tf.losses.huber_loss(labels=self.target,predictions=self.Q,reduction=tf.losses.Reduction.NONE)
        self.reg_loss = tf.losses.huber_loss(labels=self.prior,predictions=self.Q,reduction=tf.losses.Reduction.NONE)


        MAIN_DQN_VARS = find_vars(name)

        var = 25.0/(grid*grid)
        self.loss = var*self.dq_loss + var*args.LAMBDA_1 * self.reg_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)

        self.update = self.optimizer.minimize(self.loss, var_list=MAIN_DQN_VARS)
        self.grad = self.optimizer.compute_gradients(self.loss)
        # self.gradient = tf.placeholder(shape=[batch_size,grid,grid,2], dtype=tf.float32)
        # self.update = self.optimizer.apply_gradients([(self.gradient,self.Q_values)], var_list=MAIN_DQN_VARS)


    def get_one_hot(self, state, action, batch_size):
        one_hot_Q = np.zeros((batch_size, self.grid, self.grid, self.n_actions))
        for i in range(batch_size):
            one_hot_Q[i, int(state[i, 0]), int(state[i, 1]),int( action[i])] = 1

        one_hot_q_values = np.zeros((batch_size, self.grid, self.grid, self.n_actions))
        for i in range(batch_size):
            one_hot_q_values[i, int(state[i, 0]), int(state[i, 1]), :] = 1
        return one_hot_Q, one_hot_q_values

    def get_q_value(self, sess):
        q_values = np.zeros((self.grid, self.grid, self.n_actions), dtype=np.float32)
        for i in range(self.grid):
            for j in range(self.grid):
                state = np.zeros((self.grid, self.grid), dtype=np.uint8)
                state[i, j] = 1
                state = np.reshape(state, (1, -1))
                value = sess.run(self.q_values, feed_dict={self.input:state})
                q_values[i, j] = value[0]
        return q_values

def learn(session, states, actions, rewards, new_states, terminal_flags, main_dqn, target_dqn, batch_size, gamma, grid=10):
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
    states = np.squeeze(states)
    new_states = np.squeeze(new_states)
    one_hot_Q, one_hot_q_values = main_dqn.get_one_hot(new_states, actions, batch_size)
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.one_hot_q_values: one_hot_q_values})
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.one_hot_q_values: one_hot_q_values})
    double_q = q_vals[range(batch_size), arg_q_max]

    prior = np.random.normal(0,grid*grid/25,batch_size)
    target_q = rewards + (gamma * double_q * (1 - terminal_flags))

    one_hot_Q, one_hot_q_values = main_dqn.get_one_hot(states, actions, batch_size)
    #print(0, main_dqn.grad)
    #print(target_q)
    #print(states, actions, rewards, terminal_flags)
    loss, loss_dq, loss_reg, _ = session.run([main_dqn.loss, main_dqn.dq_loss, main_dqn.reg_loss, main_dqn.update],
                                              feed_dict={main_dqn.target: target_q,
                                                         main_dqn.prior: prior,
                                                         main_dqn.one_hot_Q: one_hot_Q,
                                                         main_dqn.one_hot_q_values:one_hot_q_values
                                                         })
    #print(loss)
    #print(1, main_dqn.grad)
    #session.run(main_dqn.update, feed_dict={main_dqn.gradient:grad[0][0]})
    #print(2, main_dqn.grad)
    return loss,loss_dq,loss_reg


class toy_env:
    def __init__(self, grid, final_reward=10, other_reward=-0.01, min_expert_frames=512):
        self.grid = grid
        self.final_reward = final_reward
        self.other_reward = other_reward

    def reset(self):
        self.current_state = np.array([0,0])
        return self.current_state

    def step(self, action):
        grid = self.grid
        assert not (action != 0 and action != 1), "invalid action"
        state = self.current_state
        terminal = False
        state[0] = state[0] + 1
        if action == 0:
            state[1] = max(0, state[1] - 1)
            reward = 0
            if state[0] >= grid - 1:
                print("Not success 2", grid, state[0], state[1])
                terminal = True
        else:
            state[1] = min(state[1] + 1, grid - 1)
            if state[1] == grid - 1:
                reward = self.final_reward
                print("success", grid, state[0], state[1])
                terminal = True
            elif state[0] >= grid - 1:
                reward = self.other_reward/self.grid
                print("Not success 1", grid, state[0], state[1])
                terminal = True
            else:
                reward = self.other_reward/self.grid
        return state, reward, terminal

class ActionGetter:
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""

    def __init__(self, n_actions, batch_size=32,eps_initial=1, eps_final=0.1, eps_final_frame=0.01,
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
        self.batch_size = batch_size
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
        states = np.tile(state,(self.batch_size,1))
        actions = np.zeros(self.batch_size)
        one_hot_Q, one_hot_q_values = main_dqn.get_one_hot(states, actions, self.batch_size)

        result, q_vals = session.run([main_dqn.best_action, main_dqn.q_values], feed_dict={main_dqn.one_hot_q_values: one_hot_q_values})        
        # result = result[0]
        # print(result, q_vals)
        return result[0]


def build_initial_replay_buffer(sess, env, ensemble, action_getter, max_eps, replay_buf_size, args):
    frame_num = 0
    while frame_num < replay_buf_size:
        _ = env.reset()
        for _ in range(max_eps):
            # 
            action = action_getter.get_random_action()
            # print("Action: ",action)
            # 
            next_frame, reward, terminal = env.step(action)
            #  Store transition in the replay memory
            for k in range(len(ensemble)):
                noise = np.random.normal(loc=0, scale=env.grid*env.grid/25)
                ensemble[k]["buffer"].add(obs_t=next_frame, reward=reward + noise, action=action, done=terminal)

            frame_num += 1
            if terminal:
                break

def train_step_dqfd(sess, args, env, ensemble, gradients, k,frame_num, eps_length,
                    learn, action_getter,grid):
    start_time = time.time()
    episode_reward_sum = 0
    episode_length = 0
    episode_loss = []
    episode_loss_dq = []
    episode_loss_reg = []
    REPLAY_MEMORY_START_SIZE = 32 * 200 # Number of completely random actions,
    NETW_UPDATE_FREQ = 25 * grid        # Number of chosen actions between updating the target network.
    DISCOUNT_FACTOR = args.gamma  # gamma in the Bellman equation
    UPDATE_FREQ = args.update_freq  # Every four actions a gradient descend step is performed
    BS = 32
    terminal = False
    frame = env.reset()
    MAIN_DQN = ensemble[np.random.randint(0, len(ensemble))]["main"]
    for _ in range(eps_length):
        action = action_getter.get_action(sess, frame_num, frame, MAIN_DQN)
        next_frame, reward, terminal = env.step(action)
        for i in range(k):
            noise = np.random.normal(0,grid*grid/25)
            ensemble[i]["buffer"].add(obs_t=next_frame, reward=reward+noise, action=action, done=terminal)
        episode_length += 1
        episode_reward_sum += reward
        frame_num += 1

        if frame_num % UPDATE_FREQ == 0:
            losses = []
            losses_dq = []
            losses_reg = []
            for ensemble_index in range(k):
                selected_main_dqn = ensemble[ensemble_index]["main"]

                #print(ensemble_index, selected_main_dqn.grad)
                selected_target_dqn = ensemble[ensemble_index]["target"]
                selected_buffer = ensemble[ensemble_index]["buffer"]
                generated_states, generated_actions, _, generated_rewards, generated_new_states, \
                generated_terminal_flags = selected_buffer.sample(BS,  expert=False)

                loss, loss_dq, loss_reg = learn(sess, generated_states, generated_actions, generated_rewards,generated_new_states,
                                   generated_terminal_flags, selected_main_dqn, selected_target_dqn, BS, DISCOUNT_FACTOR, grid)
                losses.append(loss)
                losses_dq.append(loss_dq)
                losses_reg.append(loss_reg)
                #gradients[ensemble_index] += grad[0][0]
                #print(len(grad), len(grad[0]), grad[0][0].shape)
                #sess.run(selected_main_dqn.update, feed_dict={selected_main_dqn.gradient:grad[0][0]})


            j = np.random.random_integers(0,k-1)
            episode_loss.append(np.mean(np.array(losses)))
            episode_loss_dq.append(np.mean(np.array(losses_dq)))
            episode_loss_reg.append(np.mean(np.array(losses_reg)))
        if frame_num % NETW_UPDATE_FREQ == 0 and frame_num > 0:
            print("UPDATING Network ... ", frame_num,"...")
            for ensemble_index in range(k):
                ensemble[ensemble_index]["updater"].update_networks(sess)
        if terminal:
            break

    return episode_reward_sum, episode_length, np.mean(episode_loss), time.time() - start_time,\
           gradients,np.mean(episode_loss_dq),np.mean(episode_loss_reg)

def eval_env(sess, args, env, ensemble, frame_num, eps_length, action_getter,grid, pretrain=False, index=-1):
    if index == -1:
        MAIN_DQN = ensemble[np.random.randint(0, len(ensemble))]["main"]
    else:
        MAIN_DQN = ensemble[index]["main"]
    episode_reward_sum = 0
    episode_len = 0
    eval_pos = []
    next_frame = env.reset()
    eval_pos.append(np.argmax(next_frame))
    for _ in range(eps_length):
        action = action_getter.get_action(sess, frame_num, next_frame, MAIN_DQN, evaluation=True)
        next_frame, reward, terminal = env.step(action)
        episode_reward_sum += reward
        episode_len += 1
        eval_pos.append(np.argmax(next_frame))
        if terminal:
            break
    return episode_reward_sum, episode_len, eval_pos

def train(priority=True,k=50):
    args = utils.argsparser()
    grid = args.grid_size
    name = args.agent
    tf.random.set_random_seed(args.seed)
    np.random.seed(args.seed)

    MAX_EPISODE_LENGTH = grid - 1
    BS = 32
    EVAL_FREQUENCY = args.eval_freq  # Number of frames the agent sees between evaluations
    NETW_UPDATE_FREQ = 25 * grid        # Number of chosen actions between updating the target network.

    REPLAY_MEMORY_START_SIZE = BS * 2  # Number of completely random actions,
    # before the agent starts learning
    MAX_FRAMES = 2**19 # Total number of frames the agent sees
    MEMORY_SIZE = BS * 40000
    # main DQN and target DQN networks:

    ensemble = []
    for i in range(k):
        print("Making network ", i)
        with tf.variable_scope('mainDQN_' + str(i)):
            MAIN_DQN = DQN(args, BS, 2, agent=name, grid=grid, name="mainDQN_" + str(i))
        with tf.variable_scope('targetDQN_' + str(i)):
            TARGET_DQN = DQN(args, BS, 2, agent=name, grid=grid, name="targetDQN_" + str(i))
        MAIN_DQN_VARS = find_vars("mainDQN_" + str(i))
        TARGET_DQN_VARS = find_vars("targetDQN_" + str(i))
        network_updater = utils.TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
        my_replay_memory = PriorityBuffer.ReplayBuffer(MEMORY_SIZE, state_shape=[2],agent_history_length=1,agent=name, batch_size=BS)
        ensemble.append({"main":MAIN_DQN, "target":TARGET_DQN, "updater":network_updater,"buffer":my_replay_memory, "idx":i})

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    action_getter = ActionGetter(2,replay_memory_start_size=REPLAY_MEMORY_START_SIZE,max_frames=MAX_FRAMES,eps_initial=0)
    sess = tf.Session(config=config)
    sess.run(init)
    # saver.restore(sess, "../models/" + name + "/" + args.env_id + "/"  + "model-" + str(5614555))

    eps_number = 0
    frame_number = 0


    tflogger = tensorflowboard_logger(
        "./" + args.log_dir + "/" + "rlsvi" + "/" + name + "_" + str(args.lr) +"_"+str(args.LAMBDA_1),
        sess, args)

    env = toy_env(grid)
    print_iter = 25
    last_eval = 0
    initial_time = time.time()
    build_initial_replay_buffer(sess, env, ensemble, action_getter, MAX_EPISODE_LENGTH, REPLAY_MEMORY_START_SIZE, args)
    
    V = env.final_reward * args.gamma ** (grid - 1) - 0.01 * (1 - args.gamma ** (grid - 1)) / (1 - args.gamma)

    gradients = [np.zeros((BS,grid,grid,2)) for i in range(k)]
    while frame_number < MAX_FRAMES:
        eps_rw, eps_len, eps_loss, eps_time, gradients,eps_dq,eps_reg = train_step_dqfd(
            sess, args, env, ensemble, gradients,k, frame_number,
            MAX_EPISODE_LENGTH, learn, action_getter, grid)

        frame_number += eps_len
        eps_number += 1
        last_eval += eps_len

        reward_list = []
        regret_list = []
        for index in range(len(ensemble)):
            eps_rw, _, _ = eval_env(sess, args, env, ensemble, frame_number, MAX_EPISODE_LENGTH, action_getter, grid, index=index)
            
            # q_values = bootstrap_dqns[index]["main"].get_q_value(sess)
            # print("q_mean", np.mean(q_values))
            # regret_list.append(compute_regret(q_values, grid, args.gamma, V, final_reward=1))
            reward_list.append(eps_rw)
        print("------------------")
        print(reward_list)
        # print(regret_list)
        print(eps_number, np.mean(reward_list[-50:]), np.mean(eps_loss), frame_number)
        if np.mean(reward_list[-50:]) > 0.1 and eps_number > 50//len(ensemble):
            return frame_number

        if eps_number % print_iter == 0:
            tflogger.log_scalar("Episode/Reward", eps_rw, eps_number)
            tflogger.log_scalar("Episode/Length", eps_len, eps_number)
            tflogger.log_scalar("Episode/Loss/Total", eps_loss, eps_number)
            tflogger.log_scalar("DQ/Loss/Total", eps_dq, eps_number)
            tflogger.log_scalar("REG/Loss/Total", eps_reg, eps_number)
            tflogger.log_scalar("Episode/Time", eps_time, eps_number)

    return -1


print("Done exploration: ",train())