import numpy as np
import gym
from skimage.transform import resize
import pybulletgym
import imageio
import os

import argparse
def argsparser():
    parser = argparse.ArgumentParser("Implementation")
    parser.add_argument('--eval_mode', help='evaluation mode, human or rgb_array', default="human", type=str)
    parser.add_argument('--eval_random', help='evaluate -> randomly get actions or not', default=True, type=bool)
    parser.add_argument('--lr', help='learning rate', default=3e-4)
    parser.add_argument('--batch_size', help='size of minibatch for minibatch-SGD', default=100)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument('--action_range', type=float, help='range of actions -> [-1, 1] * range', default=2)

    parser.add_argument('--env_id', help='choose the gym env', default='HopperPyBulletEnv-v0')
    parser.add_argument('--seed', help='random seed for repeatability', default=0)
    parser.add_argument('--max_episode_length', help='maximum length for a single run', default=1000)
    parser.add_argument("--max_timesteps", default=1e8, type=float)
    parser.add_argument("--eval_freq", default=100, type=float)
    parser.add_argument("--checkpoint_freq", default=2e5, type=float)
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='logs')
    parser.add_argument('--gif_dir', help='the directory to save GIFs file', default='gifs')
    parser.add_argument('--custom_id', help='user defined model name', default='default')
    return parser.parse_args()


def generate_gif(frame_number, frames_for_gif, reward, path):
    """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """
    imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}',
                    frames_for_gif, duration=1 / 30)

class policy:
    def __init__(self, args, action_space, obs_space):
        self.args = args
        self.action_space = action_space
        self.obs_space = obs_space
        #Implement here
        print("TBD")

    def get_random_action(self, batch_size):
        #for debugging or e-greedy
        return np.tanh(np.random.normal(0, 1, size=(batch_size, self.action_space.shape[0]))) * self.args.action_range

    def get_action(self, ob):
        #Implement, choose what arguments are needed
        print("TBD")

def evaluate(args, policy_network, env, train_step, random_sampling=True, mode="rgb_array"): #Note the mode can be rgb_array -> without gui, or human -> with gui
    max_eps_len = args.max_episode_length
    frames_list = []

    eps_reward = 0
    eps_len = 0
    if mode == "human":
        env.render(mode="human")
    current_state = env.reset()
    for i in range(max_eps_len):
        if random_sampling:
            action = policy_network.get_random_action(1)
        else:
            action = policy_network.get_action()

        action = action[0]
        next_frame, reward, terminal, _ = env.step(action)
        eps_reward += reward
        eps_len += 1

        rendered_frame = env.render(mode="rgb_array")
        frames_list.append(rendered_frame)

        if terminal:
            break;

    generate_gif(train_step, frames_list, int(np.round(eps_reward)), args.gif_dir + "/gif_")
    return eps_reward, eps_len

def train():
    args = argsparser()
    #set seed
    np.random.seed(args.seed)

    #make env
    env = gym.make(args.env_id)
    ob_space = env.observation_space
    ac_space = env.action_space

    #build network
    policy_network = policy(args, ac_space, ob_space)
    train_step_since_eval = 0
    train_step_since_checkpoint = 0


    #set the directories
    args.log_dir = "./" + args.log_dir + "/" + args.env_id  + "_seed_" + str(args.seed) + "_" + args.custom_id
    args.gif_dir = "./" + args.gif_dir + "/" + args.env_id  + "_seed_" + str(args.seed) + "_" + args.custom_id
    args.checkpoint_dir = "./" + args.checkpoint_dir + "/" + args.env_id  + "_seed_" + str(args.seed) + "_" + args.custom_id

    #make them if they do not already exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.gif_dir):
        os.makedirs(args.gif_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    #train loop
    for train_step in range(int(args.max_timesteps)):
        train_step_since_eval += 1
        train_step_since_checkpoint += 1

        if train_step_since_eval > args.eval_freq:
            train_step_since_eval = 0
            # Note random sampling should be set to false when you start actually training
            eval_rew, eval_len = evaluate(args, policy_network, env, train_step, random_sampling=args.eval_random,
                                          mode=args.eval_mode)
            print("eval rew:", eval_rew, "eval len: ", eval_len)

        if train_step_since_checkpoint > args.eval_freq:
            train_step_since_checkpoint = 0
            #save model here
            print("TBD")

if __name__ == "__main__":
    train()