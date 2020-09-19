import tensorflow as tf
import numpy as np
import time
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')
import utils
from tensorboard_logger import tensorflowboard_logger

def compute_regret(Q_value,grid,final_reward,gamma):
    pi = np.argmax(Q_value,axis=2)
    P = np.zeros((grid * grid, grid * grid))
    R = np.zeros((grid,grid))
    for i in range(grid-1):
        for j in range(grid):
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
    for j in range(grid):
        P[(grid-1)*grid+j,(grid-1)*grid+j]=1
    R = np.flatten(R)
    Q = np.matmul(np.linalg.inv(np.eye(grid*grid)-gamma*P),R)
    return Q[0,pi[0,0]]

def eps_action(action,frame_number,grid):
    if frame_number < grid*grid:
        eps=1
    elif frame_number<grid**3/2:
        slope = -0.9/(grid**3/2-grid*grid)
        intercept = 1-slope*grid*grid
        eps = slope*frame_number+intercept
    else:
        slope = -0.1/(2**grid-grid**3/2)
        intercept = 0.1-slope*grid**3/2
        eps = slope * frame_number + intercept
    if np.random.uniform(0, 1) < eps:
        p = np.random.uniform(0,1)
        if p<0.5:
            return 0
        else:
            return 1
    return action


def train( priority=True,grid=10):
    args = utils.argsparser()
    name = args.agent
    tf.random.set_random_seed(args.seed)
    np.random.seed(args.seed)

    horizon = grid
    MAX_FRAMES = 6**grid
    final_reward = 50
    gamma = 0.99
    beta = 3.0
    eta =1.0
    V = final_reward * gamma ** (grid - 2) - 0.01 * (1 - gamma ** (grid - 3)) / (1 - gamma)
    print("True value for initial state: ",V)

    # Q-value storage
    print("Agent: ", name)
    Q_value = np.zeros((grid,grid,2))
    update_count = np.zeros((grid,grid,2))
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    eps_number = 0
    frame_number = 0
    sess = tf.Session(config=config)
    sess.run(init)

    tflogger = tensorflowboard_logger(
        "./" + args.log_dir + "/" + "tabular"+"/"+"grid" + "_" + str(grid) + "_seed_" + str(args.seed),
        sess, args)

    while frame_number < MAX_FRAMES:
        terminal = False
        state = np.zeros(2,dtype=int)
        traj = np.zeros((horizon,5))
        count=0
        episode_reward_sum = 0
        episode_length = 0
        while not terminal:
            traj[count,0] = state[0]; traj[count,1]=state[1];
            Q = Q_value[state[0], state[1], :]
            action = int(np.argmax(Q))
            #epsilon-greedy action
            #action = eps_action(action,frame_number,grid)
            traj[count,2] = action
            if action == 0:
                state[0] = state[0] + 1
                state[1] = max(0, state[1] - 1)
                reward = 0
                if state[0] >= grid - 1:
                    terminal = True
            else:
                state[0] = state[0] + 1
                state[1] = min(state[1] + 1, grid - 1)
                if state[1] == grid - 1:
                    reward = final_reward
                    terminal = True
                elif state[0] >= grid - 1:
                    reward = -0.01
                    terminal = True
                else:
                    reward = -0.01
            traj[count,3]=reward
            traj[count,4] = terminal
            count +=1

            frame_number += 1
            episode_reward_sum += reward
            episode_length += 1
        if reward ==final_reward:
            print("reach optimal state at frame number: ",frame_number)
            return frame_number
        traj[count,0] = state[0]; traj[count,1]=state[1];
        frame_number += episode_length
        eps_number += 1

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        #update
        var = 0
        for i in reversed(range(grid-1)):
            state = traj[i,0:2].astype(dtype=int)
            next_state = traj[i+1,0:2].astype(dtype=int)
            action= int(traj[i,2])
            reward = traj[i,3]
            # print("state size: ",state.shape)
            update_count[state[0],state[1],action] +=1
            if state[0]==state[1]:
                # plus expert correction loss
                t = int(update_count[state[0],state[1],action])
                var = beta**2 *(t**3/3.0+5*t**2/2.0+37*t/6.0)/((t+2)**2*(t+3)**2)
                Q = Q_value[state[0], state[1], :]
                prob = softmax(eta*Q)
                expert_correction = eta*var*(action-prob[action])
                target_q = reward + gamma * np.max(Q_value[next_state[0], next_state[1]]) + expert_correction
                alpha = beta / (beta + update_count[state[0], state[1], action])
                Q_value[state[0], state[1], action] = (1 - alpha) * Q_value[state[0], state[1], action] + alpha * target_q
            else:
                target_q = reward+gamma*np.max(Q_value[next_state[0],next_state[1]])
                alpha = beta/(beta+update_count[state[0],state[1],action])
                Q_value[state[0],state[1],action] = (1-alpha) * Q_value[state[0],state[1],action] + alpha * target_q

        tflogger.log_scalar("Episode/Reward", episode_reward_sum, frame_number)
        tflogger.log_scalar("Episode/Exploration", var, frame_number)
        tflogger.log_scalar("Q_value of initial state", Q_value[0,0,1],frame_number)

        tflogger.log_scalar("Total Episodes", eps_number, frame_number)

        if eps_number%5==0:
            state = np.zeros(2, dtype=int)
            for _ in range(horizon):
                Q = Q_value[state[0], state[1], :]
                action = int(np.argmax(Q))
                if action == 0:
                    state[0] = state[0] + 1
                    state[1] = max(0, state[1] - 1)
                    reward = 0
                    if state[0] >= grid - 1:
                        terminal = True
                else:
                    state[0] = state[0] + 1
                    state[1] = min(state[1] + 1, grid - 1)
                    if state[1] == grid - 1:
                        reward = final_reward
                        terminal = True
                    elif state[0] >= grid - 1:
                        reward = -0.01
                        terminal = True
                    else:
                        reward = -0.01
                if terminal:
                    eval_reward = reward
            tflogger.log_scalar("Evaluation/Reward", eval_reward, frame_number)

import matplotlib.pyplot as plt
list = []
for grid in range(5,129):
    num = train(grid = grid)
    list.append(num)
plt.plot(range(5,129),list)
plt.savefig('toy_explor')
