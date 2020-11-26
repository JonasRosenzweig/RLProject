import gym
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import load_model


class DQN:
    def __init__(self, env, lr, gamma, eps, eps_decay):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.action_space_dim = self.action_space.n
        self.observation_space_dim = self.observation_space.shape[0]
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.rewards = []
        self.model = self.initialize_model()

    def initialize_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim = self.observation_space_dim, activation=relu))
        model.add(Dense(32, activation=relu))
        model.add(Dense(self.action_space_dim, activation=linear))
        model.compile(loss=mean_squared_error, optimizer=Adam(lr=self.lr))
        print(model.summary())
        return model

    def get_action(self, state):
        if np.random.rand() < self.eps:
            return random.randrange(self.action_space_dim)
        predicted_actions = self.model.predict(state)
        return np.argmax(predicted_actions[0])

    def train(self, episodes = 2000):
        for ep in range(episodes):
            state = env.reset()
            ep_reward = 0
            steps = 1000
            state = np.reshape(state, [1, self.observation_space_dim])
            for step in range(steps):
                # env.render()
                # time.sleep(0.0003)
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, self.observation_space_dim])
                ep_reward += reward
                state = next_state
                if done: 
                    break

    def save(self, name):
        self.model.save(name)

    def test_trained_model(self, trained_model, num_epochs=1):
        for epoch in range(num_epochs):
            epoch_rewards = []
            rewards = []
            episodes = 100
            env = gym.make("LunarLander-v2")
            steps = 1000

            for episode in range(episodes):
                trained_state = env.reset()
                ep_reward = 0
                observation_space_dim = env.observation_space.shape[0]
                trained_state = np.reshape(trained_state, [1, observation_space_dim])
                for step in range(steps):
                    env.render()
                    time.sleep(0.0003)
                    trained_action = np.argmax(trained_model.predict(trained_state)[0])
                    next_state, reward, done, info = env.step(trained_action)
                    next_state = np.reshape(next_state, [1, observation_space_dim])
                    trained_state = next_state
                    ep_reward += reward
                    if done:
                        break
                rewards.append(ep_reward)
                print("Episode: {}, total_reward: {:.2f}, last_step_reward: {:.3f}".format(episode, ep_reward, reward))
            epoch_rewards.append(rewards)
            env.close()

        for n, rewards in enumerate(epoch_rewards):
            x = range(len(rewards))
            cumsum = np.cumsum(rewards)
            avgs = [cumsum[ep]/(ep+1) if ep<100 else (cumsum[ep]-cumsum[ep-100])/100 for ep in x]
            plt.plot(x, avgs)
        plt.title("Agent Performance")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    lr = 0.001
    eps = 1.0
    eps_decay = 0.995
    gamma = 0.99
    training_episodes = 2000
    model = DQN(env, lr, gamma, eps, eps_decay)
    model.train(training_episodes)
    save_dir = "simpleDQN_"
    model.save(save_dir + "trained_model.h5")
    trained_model = load_model(save_dir + "trained_model.h5")







        
