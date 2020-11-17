import tensorflow as tf
import numpy as np
import time
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')
import utils
from tensorboard_logger import tensorflowboard_logger
from ez_greedy import ez_action_sampling

def compute_regret(Q_value,grid,final_reward,gamma):
    pi = np.argmax(Q_value,axis=2)
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
    for j in range(grid):
        P[(grid-1)*grid+j,(grid-1)*grid+j]=1
    R = np.ndarray.flatten(R)
    Q = np.matmul(np.linalg.inv(np.eye(grid*grid)-gamma*P),R)
    return Q[0]

def eps_action(action,frame_number,grid):
    c = grid/2
    if frame_number < 2**c:
        eps=1
    elif frame_number<4**grid:
        slope = -0.6/(4**grid-2**c)
        intercept = 1-slope*2**c
        eps = slope*frame_number+intercept
    else:
        slope = -0.4/(6**grid-4**grid)
        intercept = 0.4-slope*4**grid
        eps = slope * frame_number + intercept
    if np.random.uniform(0, 1) < eps:
        p = np.random.uniform(0,1)
        if p<0.5:
            return 0
        else:
            return 1
    return action


def train(grid=10,eps=True,seed =0 ):
    print(grid,":::",eps,":::")
    args = utils.argsparser()
    tf.random.set_random_seed(seed)
    np.random.seed(seed)

    horizon = grid
    MAX_FRAMES = 6**grid
    final_reward = 50
    gamma = 0.99
    beta = 4.0
    eta =1.0
    V = final_reward * gamma ** (grid - 2) - 0.01 * (1 - gamma ** (grid - 3)) / (1 - gamma)
    print("True value for initial state: ",V)

    # Q-value storage
    Q_value = np.zeros((grid,grid,2))
    update_count = np.zeros((grid,grid,2))
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    eps_number = 0
    frame_number = 0
    regret = []
    sess = tf.Session(config=config)
    sess.run(init)

    while frame_number < MAX_FRAMES:
        terminal = False
        state = np.zeros(2,dtype=int)
        traj = np.zeros((horizon,5))
        count=0
        episode_reward_sum = 0
        episode_length = 0
        duration_left = 0
        past_action = -1
        while not terminal:
            traj[count,0] = state[0]; traj[count,1]=state[1];
            Q = Q_value[state[0], state[1], :]
            action = int(np.argmax(Q))
            #epsilon-greedy action
            if eps:
                # action = eps_action(action,frame_number,grid)
                past_action, duration_left = ez_action_sampling(mu=2, n_actions=2, past_action=past_action, duration_left=duration_left)
                action = past_action
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
            for i in range(len(regret) - 1):
                regret[i + 1] += regret[i]
            return frame_number,regret
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
            if state[0]==state[1] and state[0]<grid and not eps:
                # plus expert correction loss
                t = int(update_count[state[0],state[1],action])
                var = (beta**2 +4*t)/(t+beta)**2
                Q = Q_value[state[0], state[1], :]
                prob = softmax(eta*Q)
                expert_correction = eta*var*(action-prob[action])
                target_q = reward + gamma * np.max(Q_value[next_state[0], next_state[1]]) + expert_correction
                alpha = 1.0 / (beta + update_count[state[0], state[1], action])
                Q_value[state[0], state[1], action] = (1 - alpha) * Q_value[state[0], state[1], action] + alpha * target_q
            else:
                target_q = reward+gamma*np.max(Q_value[next_state[0],next_state[1]])
                alpha = beta/(beta+update_count[state[0],state[1],action])
                Q_value[state[0], state[1], action] = (1 - alpha) * Q_value[state[0], state[1], action] + alpha * target_q

        regret.append(V - compute_regret(Q_value, grid, final_reward, gamma))
        print(V - compute_regret(Q_value, grid, final_reward, gamma))

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
    return -1,[]

train(grid=20, eps=False,seed=1)
# import matplotlib.pyplot as plt
# N = 18
# M = 20
# num_seed = 5
# eps = np.zeros((N-5,num_seed))
# expert_frame = []
# exponential = []
# for grid in range(5,15):
#     exponential.append(2 ** grid)
# for seed in range(num_seed):
#     eps_frame = []
#     for grid in range(5,N):
#         eps_num,eps_reg = train(grid = grid,eps=True,seed=seed)
#         eps_frame.append(eps_num/grid)
#
#     eps[:,seed] = np.array(eps_frame)
#
# for grid in range(5,M):
#     num, reg = train(grid=grid, eps=False,seed=seed)
#     expert_frame.append(num/grid)
# exponential = np.array(exponential)
# eps_frame = np.mean(eps,axis=1)
# eps_err = np.std(eps,axis=1)
# expert_frame = np.array(expert_frame)
#
# plt.plot(range(5,15), exponential,label='dithering')
# plt.plot(range(5,N),eps_frame,label='epsilon_greedy')
# plt.plot(range(5,M),expert_frame,label='expert')
# plt.legend()
# plt.savefig('toy_explor')
# plt.close()

# num_seed=3
# for seed in range(num_seed):
#     grid = 10
#     _, eps_reg = train(grid = grid,eps=True,seed=seed)
#     plt.plot(range(len(eps_reg)), eps_reg, label='epsilon_greedy_'+str(seed))
#
# _, reg = train(grid=grid, eps=False,seed=seed)
#
# plt.plot(range(len(reg)),reg,label='expert')
# # plt.plot(range(len(reg)),range(40,40+len(reg)),label='linear')
# plt.legend()
# plt.savefig("toy_regret")
