import gym
import random
import numpy as np
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import load_model


class DQN:
    def __init__(self, env, lr, eps, eps_decay):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.action_space_dim = self.action_space.n
        self.observation_space_dim = self.observation_space.shape[0]
        self.lr = lr
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = 0.01
        self.rewards = []
        self.training_frame_count = 0
        self.model = self.initialize_model()

    def initialize_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim = self.observation_space_dim, activation=relu))
        model.add(Dense(16, activation=relu))
        model.add(Dense(self.action_space_dim, activation=linear))
        model.compile(loss=mean_squared_error, optimizer=Adam(lr=self.lr))
        print(model.summary())
        return model

    def get_action(self, state):
        if np.random.rand() < self.eps:
            return random.randrange(self.action_space_dim)
        predicted_actions = self.model.predict(state)
        return np.argmax(predicted_actions[0])

    def train(self, episodes):
        for episode in range(episodes):
            steps = 1000
            state = env.reset()
            ep_reward = 0
            state = np.reshape(state, [1, self.observation_space_dim])
            episode_frame_count = 0
            for step in range(steps):
                #env.render()
                #time.sleep(0.0003)
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, self.observation_space_dim])
                ep_reward += reward
                state = next_state
                self.training_frame_count += 1
                episode_frame_count += 1
                if done: 
                    break
                if self.training_frame_count == 2000000:
                    break
            if self.eps > self.eps_min:
                self.eps *= self.eps_decay
                average_reward = np.mean(self.rewards[-100:])
            self.rewards.append(ep_reward)
            print("Tr Ep: {}, Ep Reward: {:.2f}, Last Reward: {:.2f}, Total Frames: {}, Frames: {}, Avg Reward: {:.2f}, Eps: {:.2f}".format(episode,
                                                                                                                                 ep_reward, 
                                                                                                                                 reward, 
                                                                                                                                 self.training_frame_count, 
                                                                                                                                 episode_frame_count, 
                                                                                                                                 average_reward,
                                                                                                                                 self.eps))
            
            
            

    def save(self, name):
        self.model.save(name)

    def test_trained_model(self, trained_model, num_episodes):
        for episode in range(num_episodes):
            rewards = []
            steps = 1000
            trained_state = env.reset()
            ep_reward = 0
            observation_space_dim = env.observation_space.shape[0]
            trained_state = np.reshape(trained_state, [1, observation_space_dim])
            for step in range(steps):
                #env.render()
                #time.sleep(0.0003)
                trained_action = np.argmax(trained_model.predict(trained_state)[0])
                next_state, reward, done, info = env.step(trained_action)
                next_state = np.reshape(next_state, [1, observation_space_dim])
                trained_state = next_state
                ep_reward += reward
                if done:
                    break
                average_reward = np.mean(rewards[-100:])
            rewards.append(ep_reward)
            print("Tr Ep: {}, Ep Reward: {:.2f}, Last Reward: {:.2f}, Avg Reward: {:.2f}".format(episode,
                                                                                             ep_reward, 
                                                                                             reward, 
                                                                                             average_reward))

            

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env.seed(21)
    np.random.seed(21)
    lr = 0.001
    eps = 1.0
    eps_decay = 0.995
    training_episodes = 200
    model = DQN(env, lr, eps, eps_decay)
    model.train(training_episodes)
    model.save("simpleDQN_trained_model.h5")
    trained_model = load_model("simpleDQN_trained_model.h5")
    model.test_trained_model(trained_model, num_episodes=10)







        
