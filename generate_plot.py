import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# fig, axs = plt.subplots(2, 2)

# data = np.load("D:\Downloads\\breakout_dqn\\aggregate.npy")
# x = data[:, 0]
# y = data[:, 1]

# ysmoothed = gaussian_filter1d(y, sigma=3)
# axs[0, 0].plot(x, y, color="blue", alpha=0.1)
# axs[0, 0].plot(x, ysmoothed,  color="blue", label="DQN", alpha=0.8, linewidth=5)


# data = np.load("D:\Downloads\\breakout_dqfd\\aggregate.npy")
# x = data[:, 0]
# y = data[:, 1]

# ysmoothed = gaussian_filter1d(y, sigma=3)
# axs[0, 0].plot(x, y, color="orange", alpha=0.1)
# axs[0, 0].plot(x, ysmoothed,  color="orange", label="DQfD", alpha=0.8, linewidth=5)


# data = np.load("D:\Downloads\\breakout_bqfd\\aggregate.npy")
# x = data[:, 0]
# y = data[:, 1]

# ysmoothed = gaussian_filter1d(y, sigma=3)
# axs[0, 0].plot(x, y, color="red", alpha=0.1)
# axs[0, 0].plot(x, ysmoothed,  color="red", label="BQfD", alpha=0.8, linewidth=5)
# axs[0, 0].set_xlabel('Frame Number', fontsize=40)
# axs[0, 0].set_ylabel('Reward', fontsize=40)
# axs[0, 0].legend(loc='upper left', fontsize=40)
# axs[0, 0].tick_params("x", labelsize=40)
# axs[0, 0].tick_params("y", labelsize=40)
# axs[0, 0].set_title("(a) Breakout", fontsize=50)
# axs[0, 0].xaxis.get_offset_text().set_fontsize(40)





# data = np.load("D:\Downloads\\seaquest_dqn\\aggregate.npy")
# x = data[:, 0]
# y = data[:, 1]

# ysmoothed = gaussian_filter1d(y, sigma=3)
# axs[0, 1].plot(x, y, color="blue", alpha=0.1)
# axs[0, 1].plot(x, ysmoothed,  color="blue", label="DQN", alpha=0.8, linewidth=5)


# data = np.load("D:\Downloads\\seaquest_dqfd\\aggregate.npy")
# x = data[:, 0]
# y = data[:, 1]

# ysmoothed = gaussian_filter1d(y, sigma=3)
# axs[0, 1].plot(x, y, color="orange", alpha=0.1)
# axs[0, 1].plot(x, ysmoothed,  color="orange", label="DQfD", alpha=0.8, linewidth=5)
# data = np.load("D:\Downloads\\seaquest_bqfd\\aggregate.npy")
# x = data[:, 0]
# y = data[:, 1]

# ysmoothed = gaussian_filter1d(y, sigma=3)
# axs[0, 1].plot(x, y, color="red", alpha=0.1)
# axs[0, 1].plot(x, ysmoothed,  color="red", label="BQfD", alpha=0.8, linewidth=5)
# axs[0, 1].set_xlabel('Frame Number', fontsize=40)
# axs[0, 1].set_ylabel('Reward', fontsize=40)
# axs[0, 1].legend(loc='upper left', fontsize=40)
# axs[0, 1].tick_params("x", labelsize=40)
# axs[0, 1].tick_params("y", labelsize=40)
# axs[0, 1].set_title("(c) SeaQuest", fontsize=50)
# axs[0, 1].xaxis.get_offset_text().set_fontsize(40)




# data = np.load("D:\Downloads\\freeway_dqn\\aggregate.npy")
# x = data[:, 0]
# y = data[:, 1]

# ysmoothed = gaussian_filter1d(y, sigma=3)
# axs[1, 1].plot(x, y, color="blue", alpha=0.1)
# axs[1, 1].plot(x, ysmoothed,  color="blue", label="DQN", alpha=0.8, linewidth=5)


# data = np.load("D:\Downloads\\freeway_dqfd\\aggregate.npy")
# x = data[:, 0]
# y = data[:, 1]

# ysmoothed = gaussian_filter1d(y, sigma=3)
# axs[1, 1].plot(x, y, color="orange", alpha=0.1)
# axs[1, 1].plot(x, ysmoothed,  color="orange", label="DQfD", alpha=0.8, linewidth=5)
# data = np.load("D:\Downloads\\freeway_bqfd\\aggregate.npy")
# x = data[:, 0]
# y = data[:, 1]

# ysmoothed = gaussian_filter1d(y, sigma=3)
# axs[1, 1].plot(x, y, color="red", alpha=0.1)
# axs[1, 1].plot(x, ysmoothed,  color="red", label="BQfD", alpha=0.8, linewidth=5)
# axs[1, 1].set_xlabel('Frame Number', fontsize=40)
# axs[1, 1].set_ylabel('Reward', fontsize=40)
# axs[1, 1].legend(loc='lower right', fontsize=40)
# axs[1, 1].tick_params("x", labelsize=40)
# axs[1, 1].tick_params("y", labelsize=40)
# axs[1, 1].set_title("(d) Freeway", fontsize=50)
# axs[1, 1].xaxis.get_offset_text().set_fontsize(40)



# data = np.load("D:\Downloads\\chopper_dqn\\aggregate.npy")
# x = data[:, 0]
# y = data[:, 1]

# ysmoothed = gaussian_filter1d(y, sigma=3)
# axs[1, 0].plot(x, y, color="blue", alpha=0.1)
# axs[1, 0].plot(x, ysmoothed,  color="blue", label="DQN", alpha=0.8, linewidth=5)


# data = np.load("D:\Downloads\\chopper_dqfd\\aggregate.npy")
# x = data[:, 0]
# y = data[:, 1]

# ysmoothed = gaussian_filter1d(y, sigma=3)
# axs[1, 0].plot(x, y, color="orange", alpha=0.1)
# axs[1, 0].plot(x, ysmoothed,  color="orange", label="DQfD", alpha=0.8, linewidth=5)
# data = np.load("D:\Downloads\\chopper_bqfd\\aggregate.npy")
# x = data[:, 0]
# y = data[:, 1]

# ysmoothed = gaussian_filter1d(y, sigma=3)
# axs[1, 0].plot(x, y, color="red", alpha=0.1)
# axs[1, 0].plot(x, ysmoothed,  color="red", label="BQfD", alpha=0.8, linewidth=5)
# axs[1, 0].set_xlabel('Frame Number', fontsize=40)
# axs[1, 0].set_ylabel('Reward', fontsize=40)
# axs[1, 0].legend(loc='upper left', fontsize=40)
# axs[1, 0].tick_params("x", labelsize=40)
# axs[1, 0].tick_params("y", labelsize=40)
# axs[1, 0].set_title("(b) ChopperCommand", fontsize=50)

# axs[1, 0].xaxis.get_offset_text().set_fontsize(40)


# fig.set_size_inches(50, 26, forward=True)
# fig.savefig("figures/atari-result.png")







fig, axs = plt.subplots(1, 2)
data = np.load("D:\Downloads\\pos_dqn\\aggregate.npy")
x = data[:, 0]
y = data[:, 1]
ysmoothed = gaussian_filter1d(y, sigma=3)
axs[0].plot(x, y, color="blue", alpha=0.1, linewidth=5)
axs[0].plot(x, ysmoothed,  color="blue", label="DQN", alpha=0.8, linewidth=5)


data = np.load("D:\Downloads\\pos_dqfd\\aggregate.npy")
x = data[:, 0]
y = data[:, 1]
ysmoothed = gaussian_filter1d(y, sigma=3)
axs[0].plot(x, y, color="orange", alpha=0.1, linewidth=5)
axs[0].plot(x, ysmoothed,  color="orange", label="DQfD", alpha=0.8, linewidth=5)


data = np.load("D:\Downloads\\pos_bqfd\\aggregate.npy")
x = data[:, 0]
y = data[:, 1]
ysmoothed = gaussian_filter1d(y, sigma=3)
axs[0].plot(x, y, color="red", alpha=0.1, linewidth=5)
axs[0].plot(x, ysmoothed,  color="red", label="BQfD", alpha=0.8, linewidth=5)
axs[0].set_xlabel('Episode Number', fontsize=40)
axs[0].set_ylabel('Reward', fontsize=40)
axs[0].legend(loc='lower right', fontsize=40)
axs[0].tick_params("x", labelsize=40)
axs[0].tick_params("y", labelsize=40)
axs[0].set_title("(a) Treasure Environment", fontsize=50)
axs[0].xaxis.get_offset_text().set_fontsize(40)

data = np.load("D:\Downloads\\neg_dqn\\aggregate.npy")
x = data[:, 0]
y = data[:, 1]
ysmoothed = gaussian_filter1d(y, sigma=3)
axs[1].plot(x, y, color="blue", alpha=0.1, linewidth=5)
axs[1].plot(x, ysmoothed,  color="blue", label="DQN", alpha=0.8, linewidth=5)


data = np.load("D:\Downloads\\neg_dqfd\\aggregate.npy")
x = data[:, 0]
y = data[:, 1]
ysmoothed = gaussian_filter1d(y, sigma=3)
axs[1].plot(x, y, color="orange", alpha=0.1, linewidth=5)
axs[1].plot(x, ysmoothed,  color="orange", label="DQfD", alpha=0.8, linewidth=5)
data = np.load("D:\Downloads\\neg_bqfd\\aggregate.npy")
x = data[:, 0]
y = data[:, 1]

ysmoothed = gaussian_filter1d(y, sigma=3)
axs[1].plot(x, y, color="red", alpha=0.1, linewidth=5)
axs[1].plot(x, ysmoothed,  color="red", label="BQfD", alpha=0.8, linewidth=5)
axs[1].set_xlabel('Episode Number', fontsize=40)
axs[1].set_ylabel('Reward', fontsize=40)
axs[1].legend(loc='lower right', fontsize=40)
axs[1].tick_params("x", labelsize=40)
axs[1].tick_params("y", labelsize=40)
axs[1].set_title("(b) Bomb Environment", fontsize=50)
axs[1].xaxis.get_offset_text().set_fontsize(40)


fig.set_size_inches(30, 12, forward=True)
fig.savefig("figures/chain_reward.png")