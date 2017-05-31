from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
import gym
import ppaquette_gym_super_mario
from model import build_network
from train import ActingAgent
from scipy.misc import imresize
from skimage.color import rgb2gray
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Evaluation of model')
parser.add_argument('--game', default='ppaquette/SuperMarioBros-1-1-v0', help='Name of openai gym environment', dest='game')
parser.add_argument('--evaldir', default=None, help='Directory to save evaluation', dest='evaldir')
parser.add_argument('--model', help='File with weights for model', dest='model')

def main():
    args = parser.parse_args()
    env = gym.make(args.game)
    if args.evaldir:
        env.monitor.start(args.evaldir)

    agent = ActingAgent(env.action_space.num_discrete_space)
    model_file = args.model
    agent.load_net.load_weights(model_file)

    game = 1
    for _ in range(10):
        done = False
        episode_reward = 0
        noops = 0

        # init game
        observation = env.reset()
        agent.init_episode(observation)
        # play one game
        print('Game #%8d; ' % (game,), end='')
        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            if action == 0:
                noops += 1
            else:
                noops = 0
            if noops > 100:
                break
        print('Reward %4d; ' % (episode_reward,))
        game += 1

    if args.evaldir:
        env.monitor.close()


if __name__ == "__main__":
    main()
