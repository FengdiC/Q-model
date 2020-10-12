import tensorflow as tf
import numpy as np
import time
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')
import utils
from tensorboard_logger import tensorflowboard_logger

def compute_regret(Q_value,grid,final_reward,gamma):
    pi = np.zeros((grid,grid))
    for i in range(grid):
        for j in range(grid):
            Q = np.array([Q_value[i * (grid - 1) + j], Q_value[i * (grid - 1) + j + grid ** 2]])
            pi[i,j] = int(np.argmax(Q))

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
    final_reward = 1
    gamma = 0.99
    beta = 4.0
    eta =3.0
    V = final_reward * args.gamma ** (grid - 1) - 0.01/grid * (1 - args.gamma ** (grid - 1)) / (1 - args.gamma)
    print("True value for initial state: ",V)

    # Q-value storage
    Q_value = np.zeros(grid*grid*2)
    variance = np.eye(grid*grid*2)*args.var
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
        while not terminal:
            traj[count,0] = state[0]; traj[count,1]=state[1];
            Q = np.array([Q_value[state[0]*(grid-1)+ state[1]],Q_value[state[0]*(grid-1)+ state[1]+grid**2]])
            action = int(np.argmax(Q))
            #epsilon-greedy action
            if eps:
                action = eps_action(action,frame_number,grid)
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
            print("reach optimal state at frame number: ",frame_number/grid)
            for i in range(len(regret) - 1):
                regret[i + 1] += regret[i]
            return frame_number,regret
        traj[count,0] = state[0]; traj[count,1]=state[1];
        eps_number += 1


        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        #update
        for i in reversed(range(grid-1)):
            state = traj[i,0:2].astype(dtype=int)
            next_state = traj[i+1,0:2].astype(dtype=int)
            action= int(traj[i,2])
            Q = np.array([Q_value[state[0] * (grid - 1) + state[1]], Q_value[state[0] * (grid - 1) + state[1] + grid ** 2]])
            next_Q = np.array([Q_value[next_state[0]*(grid-1)+ next_state[1]],Q_value[next_state[0]*(grid-1)+ next_state[1]+grid**2]])
            next_action = np.argmax(next_Q)
            T = np.eye(grid*grid*2)
            alpha = 1.0 / (beta + update_count[state[0], state[1], action])
            T[state[0]*(grid-1)+ state[1]+action*grid**2,state[0]*(grid-1)+ state[1]+action*grid**2]= 1-alpha
            T[state[0] * (grid - 1) + state[1] + action * grid ** 2,
              next_state[0] * (grid - 1) + next_state[1] + next_action * grid ** 2] = gamma*alpha
            reward = traj[i,3]
            R = np.zeros(grid*grid*2)
            R[state[0]*(grid-1)+ state[1]+action*grid**2] = reward
            # print("state size: ",state.shape)
            update_count[state[0],state[1],action] +=1
            noise = np.zeros((grid*grid*2,grid*grid*2))
            noise[state[0]*(grid-1)+ state[1]+action*grid**2,state[0]*(grid-1)+ state[1]+action*grid**2] = alpha*args.var

            #prediction step
            Q_value = np.matmul(T,Q_value) + R*alpha
            variance = np.matmul(np.matmul(np.transpose(T),variance),T) + noise

            #correction step
            if state[0]==state[1] and state[0]<grid and not eps:
                expert = np.zeros(grid*grid*2)
                prob = softmax(eta * Q)
                expert[state[0]*(grid-1)+ state[1]] =  -eta * prob[0]
                expert[state[0] * (grid - 1) + state[1]+grid**2] = eta * (1-prob[1])
                # plus expert correction loss
                Q_value = Q_value+ np.matmul(variance,expert)
                hessian = np.zeros((grid*grid*2,grid*grid*2))
                hessian[state[0] * (grid - 1) + state[1],state[0] * (grid - 1) + state[1]] = prob[0]*(1-prob[0])
                hessian[state[0] * (grid - 1) + state[1], state[0] * (grid - 1) + state[1] +grid**2] =  -prob[0]*prob[1]
                hessian[state[0] * (grid - 1) + state[1]+grid**2, state[0] * (grid - 1) + state[1]+grid**2] = prob[1]*(1 - prob[1])
                hessian[state[0] * (grid - 1) + state[1]+ grid ** 2, state[0] * (grid - 1) + state[1] ] = -prob[0] * prob[1]
                variance = np.linalg.inv(np.linalg.inv(variance)+hessian)

        regret.append(V - compute_regret(Q_value, grid, final_reward, gamma))
        print(episode_reward_sum,":::",V - compute_regret(Q_value, grid, final_reward, gamma),":::",V)
    return -1,[]

train(grid=12, eps=False,seed=1)
import matplotlib.pyplot as plt
from toy import train
for grid in range(N,M):
    num = train(grid=grid, eps=False,seed=seed)
    expert_frame.append(num)
boot_eps = np.mean(boot_eps,axis=1)
expert_eps = np.array(expert_frame)

plt.plot(range(N,M),boot_eps,label='bootstrapped DQN')
plt.plot(range(N,M),expert_eps,label='expert')
plt.legend()
plt.savefig('toy_explor')
plt.close()
