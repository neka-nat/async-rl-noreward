from __future__ import print_function
from scipy.misc import imresize
from skimage.color import rgb2gray
from multiprocessing import *
from collections import deque
import gym
import numpy as np
import h5py
import argparse
import ppaquette_gym_super_mario
from model import build_network
from noreward_models import build_icm_model

parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--game', default='ppaquette/SuperMarioBros-1-1-v0', help='OpenAI gym environment name', dest='game', type=str)
parser.add_argument('--processes', default=4, help='Number of processes that generate experience for agent',
                    dest='processes', type=int)
parser.add_argument('--lr', default=0.001, help='Learning rate', dest='learning_rate', type=float)
parser.add_argument('--steps', default=8000000, help='Number of frames to decay learning rate', dest='steps', type=int)
parser.add_argument('--batch_size', default=20, help='Batch size to use during training', dest='batch_size', type=int)
parser.add_argument('--swap_freq', default=100, help='Number of frames before swapping network weights',
                    dest='swap_freq', type=int)
parser.add_argument('--checkpoint', default=0, help='Frame to resume training', dest='checkpoint', type=int)
parser.add_argument('--save_freq', default=250000, help='Number of frames before saving weights', dest='save_freq',
                    type=int)
parser.add_argument('--queue_size', default=256, help='Size of queue holding agent experience', dest='queue_size',
                    type=int)
parser.add_argument('--n_step', default=5, help='Number of steps', dest='n_step', type=int)
parser.add_argument('--reward_scale', default=1., dest='reward_scale', type=float)
parser.add_argument('--beta', default=0.01, dest='beta', type=float)
parser.add_argument('--with_reward', dest='with_reward', action='store_true')
args = parser.parse_args()
screen = (42, 42)
input_depth = 1
past_range = 3
observation_shape = (input_depth * past_range,) + screen

def transform_screen(data):
    return rgb2gray(imresize(data, screen))[None, ...]

def policy_loss(adventage=0., beta=0.01):
    from keras import backend as K
    def loss(y_true, y_pred):
        return -K.sum(K.log(K.sum(y_true * y_pred, axis=-1) + K.epsilon()) * K.flatten(adventage)) + \
               beta * K.sum(y_pred * K.log(y_pred + K.epsilon()))
    return loss

def value_loss():
    from keras import backend as K
    def loss(y_true, y_pred):
        return 0.5 * K.sum(K.square(y_true - y_pred))
    return loss

class LearningAgent(object):
    def __init__(self, action_space, batch_size=32, swap_freq=200):
        from keras.optimizers import RMSprop
        _, _, self.train_net, adventage = build_network(observation_shape,
                                                        action_space.num_discrete_space)
        self.icm = build_icm_model(screen, (action_space.num_discrete_space,))

        self.train_net.compile(optimizer=RMSprop(epsilon=0.1, rho=0.99),
                               loss=[value_loss(), policy_loss(adventage, args.beta)])
        self.icm.compile(optimizer="rmsprop", loss=[lambda y_true, y_pred: y_pred, "mse"])

        self.pol_loss = deque(maxlen=25)
        self.val_loss = deque(maxlen=25)
        self.values = deque(maxlen=25)
        self.entropy = deque(maxlen=25)
        self.swap_freq = swap_freq
        self.swap_counter = self.swap_freq
        self.batch_size = batch_size
        self.unroll = np.arange(self.batch_size)
        self.targets = np.zeros((self.batch_size, action_space.num_discrete_space))
        self.counter = 0

    def learn(self, last_observations, actions, rewards, learning_rate=0.001):
        import keras.backend as K
        K.set_value(self.train_net.optimizer.lr, learning_rate)
        frames = len(last_observations)
        self.counter += frames
        values, policy = self.train_net.predict([last_observations, self.unroll])

        self.targets.fill(0.)
        adventage = rewards - values.flatten()
        self.targets[self.unroll, :] = actions.astype(np.float32)

        loss = self.train_net.train_on_batch([last_observations, adventage],
                                             [rewards, self.targets])
        loss_icm = self.icm.train_on_batch([last_observations[:, -2, ...],
                                            last_observations[:, -1, ...],
                                            actions],
                                            [np.zeros((self.batch_size,)), actions])
        entropy = np.mean(-policy * np.log(policy + 0.00000001))
        self.store_results(loss, entropy, values, loss_icm)
        self.swap_counter -= frames
        if self.swap_counter < 0:
            self.swap_counter += self.swap_freq
            return True
        return False

    def store_results(self, loss, entropy, values, loss_icm):
        self.pol_loss.append(loss[2])
        self.val_loss.append(loss[1])
        self.entropy.append(entropy)
        self.values.append(np.mean(values))
        min_val, max_val, avg_val = min(self.values), max(self.values), np.mean(self.values)
        print('\rFrames: %8d; Policy-Loss: %10.6f; Avg: %10.6f '
              '--- Value-Loss: %10.6f; Avg: %10.6f '
              '--- Entropy: %7.6f; Avg: %7.6f '
              '--- V-value; Min: %6.3f; Max: %6.3f; Avg: %6.3f'
              '--- ICM-Loss: %10.6f; Action-Loss: %10.6f' % (
                  self.counter,
                  loss[2], np.mean(self.pol_loss),
                  loss[1], np.mean(self.val_loss),
                  entropy, np.mean(self.entropy),
                  min_val, max_val, avg_val,
                  loss_icm[1], loss_icm[2]), end='')

def learn_proc(mem_queue, weight_dict):
    import os
    pid = os.getpid()
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=False,lib.cnmem=0.3,' + \
                                 'compiledir=th_comp_learn,optimizer=fast_compile'
    print(' %5d> Learning process' % (pid,))
    save_freq = args.save_freq
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    steps = args.steps

    env = gym.make(args.game)
    agent = LearningAgent(env.action_space, batch_size=args.batch_size, swap_freq=args.swap_freq)
    if checkpoint > 0:
        print(' %5d> Loading weights from file' % (pid,))
        agent.train_net.load_weights('model-%s-%d.h5' % (args.game, checkpoint,))

    print(' %5d> Setting weights in dict' % (pid,))
    weight_dict['update'] = 0
    weight_dict['weights'] = agent.train_net.get_weights()
    weight_dict['weights_icm'] = agent.icm.get_weights()

    last_obs = np.zeros((batch_size,) + observation_shape)
    actions = np.zeros((batch_size, env.action_space.num_discrete_space), dtype=np.int32)
    rewards = np.zeros(batch_size)

    idx = 0
    agent.counter = checkpoint
    save_counter = checkpoint % save_freq + save_freq
    while True:
        last_obs[idx, ...], actions[idx, ...], rewards[idx] = mem_queue.get()
        idx = (idx + 1) % batch_size
        if idx == 0:
            lr = max(0.00000001, (steps - agent.counter) / steps * learning_rate)
            updated = agent.learn(last_obs, actions, rewards, learning_rate=lr)
            if updated:
                # print(' %5d> Updating weights in dict' % (pid,))
                weight_dict['weights'] = agent.train_net.get_weights()
                weight_dict['weights_icm'] = agent.icm.get_weights()
                weight_dict['update'] += 1
        save_counter -= 1
        if save_counter < 0:
            save_counter += save_freq
            agent.train_net.save_weights('model-%s-%d.h5' % (args.game, agent.counter,), overwrite=True)


class ActingAgent(object):
    def __init__(self, num_action, n_step=8, discount=0.99):

        self.value_net, self.policy_net, self.load_net, _ = build_network(observation_shape,
                                                                          num_action)
        self.icm = build_icm_model(screen, (num_action,))

        self.value_net.compile(optimizer='rmsprop', loss='mse')
        self.policy_net.compile(optimizer='rmsprop', loss='mse')
        self.load_net.compile(optimizer='rmsprop', loss='mse', loss_weights=[0.5, 1.])  # dummy loss
        self.icm.compile(optimizer="rmsprop", loss=[lambda y_true, y_pred: y_pred, "mse"])

        self.num_action = num_action
        self.observations = np.zeros(observation_shape)
        self.last_observations = np.zeros_like(self.observations)

        self.n_step_data = deque(maxlen=n_step)
        self.n_step = n_step
        self.discount = discount

    def init_episode(self, observation):
        for _ in range(past_range):
            self.save_observation(observation)

    def reset(self):
        self.n_step_data.clear()

    def sars_data(self, action, reward, observation, terminal, mem_queue):
        self.save_observation(observation)
        reward = np.clip(reward, -1., 1.)
        # reward /= args.reward_scale

        self.n_step_data.appendleft([self.last_observations,
                                     action, reward])

        if terminal or len(self.n_step_data) >= self.n_step:
            r = 0.
            if not terminal:
                r = self.value_net.predict(self.observations[None, ...])[0]
            for i in range(len(self.n_step_data)):
                r = self.n_step_data[i][2] + self.discount * r
                mem_queue.put((self.n_step_data[i][0], self.n_step_data[i][1], r))
            self.reset()

    def choose_action(self, eps=0.1):
        if np.random.rand(1) < eps:
            action = np.random.rand(self.num_action)
        else:
            action = self.policy_net.predict(self.observations[None, ...])[0]
        action = np.round(action)
        return action.astype(np.int32)

    def save_observation(self, observation):
        self.last_observations = self.observations[...]
        self.observations = np.roll(self.observations, -input_depth, axis=0)
        self.observations[-input_depth:, ...] = transform_screen(observation)

def generate_experience_proc(mem_queue, weight_dict, no):
    import os
    pid = os.getpid()
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=True,lib.cnmem=0,' + \
                                 'compiledir=th_comp_act_' + str(no)
    print(' %5d> Process started' % (pid,))
    frames = 0
    batch_size = args.batch_size
    env = gym.make(args.game)
    agent = ActingAgent(env.action_space.num_discrete_space, n_step=args.n_step)

    if frames > 0:
        print(' %5d> Loaded weights from file' % (pid,))
        agent.load_net.load_weights('model-%s-%d.h5' % (args.game, frames))
    else:
        import time
        while 'weights' not in weight_dict:
            time.sleep(0.1)
        agent.load_net.set_weights(weight_dict['weights'])
        agent.icm.set_weights(weight_dict['weights_icm'])
        print(' %5d> Loaded weights from dict' % (pid,))

    best_score = 0
    avg_score = deque([0], maxlen=25)
    last_update = 0
    eta = 1.0 # parameter for intrinsic reward
    while True:
        done = False
        episode_reward = 0
        op_last, op_count = np.zeros(env.action_space.num_discrete_space), 0
        observation = env.reset()
        obs_last = observation.copy()
        agent.init_episode(observation)

        while not done:
            frames += 1
            action = agent.choose_action()
            observation, reward, done, _ = env.step(action)
            #env.render()
            r_in, _ = agent.icm.predict([transform_screen(obs_last),
                                         transform_screen(observation),
                                         np.array([action])])
            if args.with_reward:
                total_reward = reward + eta * r_in[0]
            else:
                total_reward = eta * r_in[0]
            episode_reward += total_reward
            best_score = max(best_score, episode_reward)
            agent.sars_data(action, total_reward, observation, done, mem_queue)
            op_count = 0 if (op_last != action).any() else op_count + 1
            done = done or op_count >= 100
            op_last = action
            obs_last = observation.copy()
            if frames % 2000 == 0:
                print(' %5d> Best: %4d; Avg: %6.2f; Max: %4d' % (
                      pid, best_score, np.mean(avg_score), np.max(avg_score)))
            if frames % batch_size == 0:
                update = weight_dict.get('update', 0)
                if update > last_update:
                    last_update = update
                    # print(' %5d> Getting weights from dict' % (pid,))
                    agent.load_net.set_weights(weight_dict['weights'])
                    agent.icm.set_weights(weight_dict['weights_icm'])
        avg_score.append(episode_reward)

def init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    manager = Manager()
    weight_dict = manager.dict()
    mem_queue = manager.Queue(args.queue_size)

    pool = Pool(args.processes + 1, init_worker)

    try:
        for i in range(args.processes):
            pool.apply_async(generate_experience_proc, (mem_queue, weight_dict, i))

        pool.apply_async(learn_proc, (mem_queue, weight_dict))

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()
