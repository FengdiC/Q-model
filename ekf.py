import tensorflow as tf
import numpy as np
import time
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')
import utils
from tensorboard_logger import tensorflowboard_logger
import matplotlib.pyplot as plt

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


def train(grid=10,eps=True ):
    print(grid,":::",eps,":::")
    MAX_FRAMES = 6000
    final_reward = 1
    gamma = 0.99
    beta = 4.0
    eta =3.0
    var = 0.015
    Q_value = np.zeros(grid*grid*2)
    Q_value[grid**2:] = 1
    V = compute_regret(Q_value, grid, final_reward, gamma)
    print("True value for initial state: ", V)

    # Q-value storage
    Q_value = np.zeros(grid*grid*2)
    variance = np.eye(grid*grid*2)*var
    update_count = np.zeros((grid,grid,2))
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    eps_number = 0
    frame_number = 0
    regret = []
    sess = tf.Session(config=config)
    sess.run(init)
    final=False

    while frame_number < MAX_FRAMES:
        terminal = False
        state = np.zeros(2,dtype=int)
        traj = np.zeros((grid,5))
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
            print("reach optimal state at frame number: ",frame_number)
            final=True
            # if frame_number>4200:
            #     for i in range(len(regret) - 1):
            #         regret[i + 1] += regret[i]
            #     return frame_number, regret
            if len(regret)>3 and np.mean(regret[-3:])<0.001 and final:
                # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
                # plt.tight_layout()
                # plt.savefig('kalman_gain')
                return eps_number
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
            update_count[state[0], state[1], action] += 1
            t= update_count[state[0],state[1],action]
            Q = np.array([Q_value[state[0] * (grid - 1) + state[1]], Q_value[state[0] * (grid - 1) + state[1] + grid ** 2]])
            next_Q = np.array([Q_value[next_state[0]*(grid-1)+ next_state[1]],Q_value[next_state[0]*(grid-1)+ next_state[1]+grid**2]])
            next_action = np.argmax(next_Q)
            T = np.eye(grid*grid*2)
            alpha = 1.0 / (beta + t)
            # alpha = 0.001
            T[state[0]*(grid-1)+ state[1]+action*grid**2,state[0]*(grid-1)+ state[1]+action*grid**2]= 1-alpha
            T[state[0] * (grid - 1) + state[1] + action * grid ** 2,
              next_state[0] * (grid - 1) + next_state[1] + next_action * grid ** 2] = gamma*alpha
            reward = traj[i,3]
            R = np.zeros(grid*grid*2)
            R[state[0]*(grid-1)+ state[1]+action*grid**2] = reward

            noise = np.zeros((grid*grid*2,grid*grid*2))
            noise[state[0]*(grid-1)+ state[1]+action*grid**2,state[0]*(grid-1)+ state[1]+action*grid**2] = alpha*var

            #prediction step
            Q_value = np.matmul(T,Q_value) + R*alpha
            variance = np.matmul(np.matmul(np.transpose(T),variance),T) + noise
            # diff = variance[state[0]*(grid-1)+ state[1]+action*grid**2,state[0]*(grid-1)+ state[1]+action*grid**2]\
            #        -(beta**2 +4*t)/(t+beta)**2
            # print("variance estimate: ",diff,":::",diff/variance[state[0]*(grid-1)+ state[1]+action*grid**2,state[0]*(grid-1)+ state[1]+action*grid**2])

            #correction step
            if state[0]==state[1] and state[0]<grid and not eps:
                expert = np.zeros(grid*grid*2)
                prob = softmax(eta * Q)
                # expert[state[0]*(grid-1)+ state[1]] =  -eta * prob[0]
                expert[state[0] * (grid - 1) + state[1]+grid**2] = eta * (1-prob[1])
                # plus expert correction loss
                correction = np.matmul(variance,expert)
                # print("state number: ",state[0],"amount: ",correction[state[0] * (grid - 1) + state[1]+grid**2]/(eta * (1-prob[1])))
                Q_value[state[0] * (grid - 1) + state[1]+grid**2] += correction[state[0] * (grid - 1) + state[1]+grid**2]

                hessian = np.zeros((grid*grid*2,grid*grid*2))
                hessian[state[0] * (grid - 1) + state[1],state[0] * (grid - 1) + state[1]] = prob[0]*(1-prob[0])
                hessian[state[0] * (grid - 1) + state[1], state[0] * (grid - 1) + state[1] +grid**2] =  -prob[0]*prob[1]
                hessian[state[0] * (grid - 1) + state[1]+grid**2, state[0] * (grid - 1) + state[1]+grid**2] = prob[1]*(1 - prob[1])
                hessian[state[0] * (grid - 1) + state[1]+ grid ** 2, state[0] * (grid - 1) + state[1] ] = -prob[0] * prob[1]

                variance = np.linalg.inv(np.linalg.inv(variance)+hessian*eta**2)

        #compute kalman gain
        if eps_number==2 or eps_number==5 or eps_number==30:
            c=np.zeros(grid)
            decay = np.zeros(grid)
            z = np.zeros((grid * grid * 2, 2*grid))
            h = np.zeros((2*grid,2*grid))
            for i in range(grid):
                Q = np.array([Q_value[i * (grid - 1) + i], Q_value[i * (grid - 1) + i + grid ** 2]])
                prob = softmax(eta * Q)
                z[i * (grid - 1) + i, 2*i] = prob[0] * (1 - prob[0]); h[2*i,2*i]= prob[0] * (1 - prob[0])
                z[i * (grid - 1) + i, 2*i+1] = -prob[0] * prob[1]; h[2*i,2*i+1] = -prob[0] * prob[1]
                z[i * (grid - 1) + i + grid ** 2, 2*i+1] = prob[1] * (1 -prob[1]); h[2*i+1,2*i+1]= prob[1] * (1 -prob[1])
                z[i* (grid - 1) +i + grid ** 2, 2*i] = -prob[0] * prob[1];h[2*i+1,2*i]= -prob[0] * prob[1]

            kalman = np.matmul(np.matmul(np.transpose(z),variance),z)+ h
            kalman = np.linalg.inv(kalman+np.eye(grid*2)*0.001)
            kalman = np.matmul(np.matmul(variance,z),kalman)
            amount = np.sum(np.absolute(kalman),axis=0)

            gain = np.zeros(grid)
            for i in range(grid):
                gain[i] = amount[2*i]+amount[2*i+1]
                c[i] = variance[i * (grid - 1) + i + grid ** 2, i * (grid - 1) + i + grid ** 2]
                t = update_count[i, i, 1]
                decay[i] = var * (beta ** 2 + 4 * t) / (t + beta) ** 2

            # # kalman gain
            # plt.subplot(211)
            # plt.plot(range(grid-1),gain[:-1],label='kalman_info_gain_'+str(eps_number))
            # plt.legend()
            # plt.subplot(212)
            # plt.plot(range(grid-1),c[:-1],label = 'GEKF_weight_'+str(eps_number))
            # plt.subplot(212)
            # plt.plot(range(grid-1), decay[:-1], label='BQfD_weight_'+str(eps_number))
            # plt.legend()

        # pi = np.zeros((grid, grid))
        # for i in range(grid):
        #     for j in range(grid):
        #         Q = np.array([Q_value[i * (grid - 1) + j], Q_value[i * (grid - 1) + j + grid ** 2]])
        #         pi[i, j] = int(np.argmax(Q))
        # correct = np.sum(pi[:, 0])
        # print("Episode: ", eps_number, "policy: ", correct,"Q-value: ",Q_value[0],":::",Q_value[grid**2])
        # print(pi[:, 0])
        # regret.append(correct)
        # regret.append(V - max(Q_value[0],Q_value[grid**2]))
        regret.append(V - compute_regret(Q_value, grid, final_reward, gamma))

        # print(V - compute_regret(Q_value, grid, final_reward, gamma))
        print(V - max(Q_value[0],Q_value[grid**2]))
        # print(episode_reward_sum,":::",V - compute_regret(Q_value, grid, final_reward, gamma),":::",V)
    return 6000/grid

# train(grid=20, eps=False)

# import matplotlib.pyplot as plt
# # num,regret_ekf = train(10,False)
# from tabular import train
# # num, regret_tabular = train(10,False,0)
# for lr in [None]:
#     regret = np.zeros((700))
#     for seed in [0,1,2,3,4]:
#         num,regret_eps = train(20,True,seed,130,lr)
#         print(len(regret_eps))
#         regret += regret_eps[:700]
#     regret /= 5.0
#     plt.plot(range(len(regret)),regret,label=str(lr))
# plt.legend()
# plt.savefig('epsilon greedy')
# plt.close()


# import matplotlib.pyplot as plt
# regret = np.zeros((650))
# num,regret_ekf = train(20,False)
# plt.plot(range(len(regret_ekf)),regret_ekf,label="full GEKF",color='g')
# from tabular import train
# num, regret_tabular = train(20,False,0)
# plt.plot(range(len(regret_tabular)),regret_tabular,label="BQfD",color='r')
# for seed in [0, 1, 2, 3, 4]:
#     num, regret_eps = train(20, True, seed, 130,None)
#     regret +=regret_eps[:650]
# regret /=5.0
# plt.plot(range(len(regret)),regret,label="EZ greedy",color='b')
# plt.legend()
# plt.savefig('regrets_')