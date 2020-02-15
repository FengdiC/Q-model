import numpy as np
import pickle

def discounted_cumulative_reward(list_rew,list_ter,n=20,discount=0.99):
    """
    Creates the TensorFlow operations for calculating the discounted cumulative rewards
    for a given sequence of rewards.
    Args:
        terminal: Terminal boolean tensor.
        reward: Reward tensor.
        discount: Discount factor.
        final_reward: Last reward value in the sequence.
    Returns:
        Discounted cumulative reward tensor.
    """
    list_rew.reverse()
    list_ter.reverse()

    for i in range(len(list_rew)):
        list_rew[i] = np.sign(list_rew[i])*np.log(1+list_rew[i])
        if i ==0:
            continue
        list_rew[i] += discount * list_rew[i-1] * (1-list_ter[i])
    list_rew.reverse()
    list_ter.reverse()
    for i in range(len(list_rew)-n):
        terminal =1
        for j in range(n):
            terminal *= (1- list_ter[i+j])
        list_rew[i] -= (discount)**n *list_rew[i+n]*terminal

    return list_rew

def averaged():
    data1 = pickle.load(open('data/average/human_SeaquestDeterministic-v4_1.pkl','rb'))['reward']
    terminal1 = pickle.load(open('data/average/human_SeaquestDeterministic-v4_1.pkl', 'rb'))['terminal']
    l1 = len(data1)
    data2 = pickle.load(open('data/average/human_SeaquestDeterministic-v4_0.pkl','rb'))['reward']
    terminal2 = pickle.load(open('data/average/human_SeaquestDeterministic-v4_0.pkl', 'rb'))['terminal']
    l2 = len(data2)
    data3 = pickle.load(open('data/average/human_SeaquestDeterministic-v4_-1.pkl','rb'))['reward']
    terminal3 = pickle.load(open('data/average/human_SeaquestDeterministic-v4_-1.pkl', 'rb'))['terminal']
    l3 = len(data3)
    data4 = pickle.load(open('data/average/human_SeaquestDeterministic-v4_-2.pkl','rb'))['reward']
    terminal4 = pickle.load(open('data/average/human_SeaquestDeterministic-v4_-2.pkl', 'rb'))['terminal']
    l4 = len(data4)
    # data5 = pickle.load(open('data/average/human_SeaquestDeterministic-v4_5.pkl','rb'))['reward'][l1+l2+l3+l4:]
    # terminal5 = pickle.load(open('data/average/human_SeaquestDeterministic-v4_5.pkl', 'rb'))['terminal'][l1+l2+l3+l4:]
    # l5 = len(data5)

    data1=discounted_cumulative_reward(data1,terminal1)
    data2=discounted_cumulative_reward(data2,terminal2)
    data3=discounted_cumulative_reward(data3,terminal3)
    data4=discounted_cumulative_reward(data4,terminal4)
    # data5 = discounted_cumulative_reward(data5, terminal5)
    length = min([l1,l2,l3,l4])
    print(length)
    print([l1,l2,l3,l4])

    reward = [None] * length

    for i in range(length):
        reward[i] = (data1[i]+data2[i]+data3[i]+data4[i])/4

    with open("reward_average_SeaquestDeterministic-v4", 'wb') as fout:
        pickle.dump(reward, fout)

def differences(discount=0.99):
    expert_data_list = pickle.load(open('data/improved_expert/human_SeaquestDeterministic-v4_1.pkl', 'rb'))
    rew = expert_data_list['reward'].copy()
    terminal = expert_data_list['terminal'].copy()
    cum_rew = discounted_cumulative_reward(rew, terminal,n=50)
    # print(rew[200:300])
    print("reward larger than 0: ",np.sum(np.array(rew)>=-0.001))
    print("cum reward larger than 0: ", np.sum(np.array(cum_rew) >0))
    diff = [0]*len(rew)

    averaged = pickle.load(open("reward_average_SeaquestDeterministic-v4", 'rb'))
    print("average larger than 0: ",np.sum(np.array(averaged) > 0))
    time = 0
    for i in range(len(rew)):
        diff[i] = cum_rew[i] - 0.8*averaged[i]

    diff = np.array(diff)
    print(len(diff))
    print(np.sum(diff > 1))
    print(np.max(diff))
    # print(diff[:100])

    # def elu(z, alpha):
    #     for i in range(z.shape[0]):
    #         if z[i]<0:
    #             z[i] = alpha * (np.exp(z[i])  - 1)
    #     return z
    # diff = elu(diff,1.0)
    # diff = np.array(diff)
    # diff = np.clip(diff, -1, 2)
    # diff *= 0.5
    expert_data_list['diff'] = diff.tolist()
    with open('human_SeaquestDeterministic-v4_1', 'wb') as fout:
        pickle.dump(expert_data_list, fout)

averaged()
differences()