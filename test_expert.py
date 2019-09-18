import numpy as np
import tensorflow as tf
import pickle
from skimage.transform import resize
import imageio

dataset = pickle.load(open('/network/home/chefengd/Documents/Q-model/data/expert_dist_dqn/SeaquestDeterministic-v4/expert_data.pkl_30','rb'))


def generate_gif(frames_for_gif):
    """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """
    #for idx, frame_idx in enumerate(frames_for_gif):
        #frames_for_gif[idx] = resize(frame_idx, (420, 320, 3),
        #                             preserve_range=True, order=0).astype(np.uint8)

    imageio.mimsave('test_expert.gif',
                    frames_for_gif, duration=1 / 5)

print("number of frames in total: ", dataset.count)

num_eps = 0
accumulated_reward = 0
for i in range(dataset.count):
    accumulated_reward += dataset.rewards[i]
    if dataset.terminal_flags[i]:
        print("{0} episode trajectory reward: ".format(num_eps),accumulated_reward)
        accumulated_reward = 0

generate_gif(dataset.frames[:20000])