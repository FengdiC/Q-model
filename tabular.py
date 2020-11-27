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
                    R[i,j] = -0.01/grid
            P[i*grid+j,l*grid+r]=1
    for j in range(grid):
        P[(grid-1)*grid+j,(grid-1)*grid+j]=1
    R = np.ndarray.flatten(R)
    Q = np.matmul(np.linalg.inv(np.eye(grid*grid)-gamma*P),R)
    return Q[0]

def eps_compute(frame_number,grid):
    c = grid/2
    if frame_number < 2**c:
        eps=1
    elif frame_number<2**c * 5:
        slope = -0.7/(2**c * 5-2**c)
        intercept = 1-slope*2**c
        eps = slope*frame_number+intercept
    else:
        eps=0.3
    return eps

def ez_action_sampling(mu, n_actions, past_action, duration_left):
    #sample randomly an action ..
    if duration_left <= 0:
        past_action = np.random.randint(0, n_actions)
        duration_left = np.random.zipf(mu)
    duration_left -= 1
    return past_action, duration_left


def train(grid=10,eps=True,seed =0 ):
    print(grid,":::",eps,":::")
    args = utils.argsparser()
    tf.random.set_random_seed(seed)
    np.random.seed(seed)

    horizon = grid
    MAX_FRAMES = 6**grid
    final_reward = 1
    gamma = 0.99
    beta = 4.0
    eta =1.2
    Q_value = np.zeros((grid, grid, 2))
    Q_value[:,:,1]=final_reward
    V = compute_regret(Q_value,grid,final_reward,gamma)
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
    final= False

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
            e = eps_compute(frame_number,grid)
            traj[count,0] = state[0]; traj[count,1]=state[1];
            Q = Q_value[state[0], state[1], :]
            action = int(np.argmax(Q))
            #epsilon-greedy action
            if eps:
                if duration_left >0:
                    past_action, duration_left = ez_action_sampling(mu=2, n_actions=2, past_action=past_action, duration_left=duration_left)
                    action = past_action
                elif np.random.uniform(0,1)< e:
                    past_action, duration_left = ez_action_sampling(mu=2, n_actions=2, past_action=past_action,duration_left=duration_left)
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
                    reward = -0.01/grid
                    terminal = True
                else:
                    reward = -0.01/grid
            traj[count,3]=reward
            traj[count,4] = terminal
            count +=1

            frame_number += 1
            episode_reward_sum += reward
            episode_length += 1
        if reward ==final_reward:
            print("reach optimal state at frame number: ",frame_number)
            final= True
            if len(regret) > 3 and regret[-2] - regret[-1] < 0.003 and eps and frame_number>2500:
                for i in range(len(regret) - 1):
                    regret[i + 1] += regret[i]
                return frame_number, regret
            if len(regret) > 3 and regret[-2] - regret[-1] < 0.003 and not eps and frame_number>4000:
                for i in range(len(regret) - 1):
                    regret[i + 1] += regret[i]
                return frame_number, regret
            # if len(regret)>3 and np.mean(regret[-3:])==0 and final:
            #     return eps_number
        traj[count,0] = state[0]; traj[count,1]=state[1]
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
                var = (0.2)*(beta**2 +4*t)/(t+beta)**2
                Q = Q_value[state[0], state[1], :]
                prob = softmax(eta*Q)
                expert_correction = eta*var*(action-prob[action])
                if eps:
                    expert_correction = 0
                target_q = reward + gamma * np.max(Q_value[next_state[0], next_state[1]]) + expert_correction
                alpha = 1.0 / (beta + update_count[state[0], state[1], action])
                Q_value[state[0], state[1], action] = (1 - alpha) * Q_value[state[0], state[1], action] + alpha * target_q
            else:
                target_q = reward+gamma*np.max(Q_value[next_state[0],next_state[1]])
                alpha = beta/(beta+update_count[state[0],state[1],action])
                Q_value[state[0], state[1], action] = (1 - alpha) * Q_value[state[0], state[1], action] + alpha * target_q

        # pi = np.argmax(Q_value, axis=2)
        # correct = np.sum(pi[:,0])
        # # print(pi[:,0])
        # regret.append(correct)
        # print("Episode: ",eps_number,"policy: ",correct)

        regret.append(V - np.max(Q_value[0,0]))
        # regret.append(V-compute_regret(Q_value,grid,final_reward,gamma))
        print("Episode: ", eps_number, "regret: ", V - np.max(Q_value[0,0]))
        # print(V - np.max(Q_value[0,0]))
    return -1,[]


# train(80,False,0)
# import matplotlib.pyplot as plt
# M=7
# N=28
#
# reach = np.zeros(N-M)
# for seed in range(5):
#     for grid in range(M,N,1):
#         print("epsilon: grid_",grid,"seed_",seed)
#         num = train(grid,True,seed)
#         reach[grid-M] += num
# reach_eps = reach/5.0
#
# reach = np.zeros(N-M)
# for grid in range(M,N,1):
#     print("tabular: grid_", grid)
#     num = train(grid,False,seed=0)
#     reach[grid-M] = num
# reach_tabular = reach
#
# from ekf import train
# reach = np.zeros(N-M)
# for grid in range(M,N,1):
#     print("ekf: grid_", grid)
#     num = train(grid,False)
#     reach[grid-M] = num
# reach_ekf = reach
#
# plt.plot(range(M,N,1),reach_eps,label='epsilon-greedy')
# plt.plot(range(M,N,1),reach_tabular,label='estimation')
# plt.plot(range(M,N,1),reach_ekf,label='GEKF')
# plt.legend()
# plt.savefig('toy_explor')