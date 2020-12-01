# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 11:21:00 2020

@author: Kata
"""
import gym
import random
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import wandb
import time
# import keras

from keras import Sequential
from keras.layers import Dense, Dropout
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import load_model

name = "LunarLander_V2_DQN_ExpReplay_{}".format(int(time.time()))        
        

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
        self.eps_min = 0.01
        self.rewards = []
        self.trained_rewards = []
        self.training_frame_count = 0
        self.replay_memory_buffer = deque(maxlen=500000)
        self.batch_size = 64
        self.model = self.initialize_model()
        self.counter = 0
        self.average_rewards = []
        self.average_rewards_trained = []
        # self.accuracy = tf.keras.metrics.CategoricalAccuracy()
        

    def initialize_model(self):
        model = Sequential()
        # The bigger the Dense layer, the higher the chance of overfitting.
        model.add(Dense(512, input_dim = self.observation_space_dim, activation=relu))
        # Dropout 0.x: Randomly deactivate x*10% of the neurons
        # Dropout will prevent overfitting, especially in Dense layers.
        # model.add(Dropout(0.2))
        model.add(Dense(256, activation=relu))
        # model.add(Dropout(0.2))
        model.add(Dense(self.action_space_dim, activation=linear))
        model.compile(loss=mean_squared_error, optimizer=Adam(lr=self.lr))
        print(model.summary())
        return model

    def get_action(self, state):
        if np.random.rand() < self.eps:
            return random.randrange(self.action_space_dim)
        predicted_actions = self.model.predict(state)
        return np.argmax(predicted_actions[0])
    
    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    def learn_and_update_weights_by_reply(self):

        # replay_memory_buffer size check
        if len(self.replay_memory_buffer) < self.batch_size or self.counter != 0:
            return

        # Early Stopping
        if np.mean(self.rewards[-10:]) > 180:
            return

        random_sample = self.get_random_sample_from_replay_mem()
        states, actions, rewards, next_states, done_list = self.get_attribues_from_sample(random_sample)
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets
        
        

        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def get_attribues_from_sample(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, done_list

    def get_random_sample_from_replay_mem(self):
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        return random_sample

    def train(self, episodes):
        wandb.init(project="LanderDQN", name="Performance")
        for episode in range(episodes):
            steps = 1000
            state = env.reset()
            ep_reward = 0
            state = np.reshape(state, [1, self.observation_space_dim])
            episode_frame_count = 0
            for step in range(steps):
                #env.render()
                #time.sleep(0.0003)
                exploit_action = self.get_action(state)
                next_state, reward, done, info = env.step(exploit_action)
                next_state = np.reshape(next_state, [1, self.observation_space_dim])
                ep_reward += reward
                self.training_frame_count += 1
                episode_frame_count += 1
                self.add_to_replay_memory(state, exploit_action, reward, next_state, done)
                state = next_state
                self.update_counter()
                self.learn_and_update_weights_by_reply()
                if done: 
                    break
            if self.eps > self.eps_min:
                self.eps *= self.eps_decay
            
            average_reward = np.mean(self.rewards[-100:])
            if average_reward > 200:
                break
            self.rewards.append(ep_reward)
            self.average_rewards.append(average_reward)
            wandb.log({'reward': average_reward, 'last reward': reward, 'epsilon': self.eps}, step=episode)
            
            print("Tr Ep: {}, Ep Reward: {:.2f}, Last Reward: {:.2f}, Total Frames: {}, Frames: {}, Avg Reward: {:.2f}, Eps: {:.2f},".format(episode,
                                                                                                                                                     ep_reward, 
                                                                                                                                                     reward, 
                                                                                                                                                     self.training_frame_count, 
                                                                                                                                                     episode_frame_count, 
                                                                                                                                                     average_reward,
                                                                                                                                                     self.eps,
                                                                                                                                                     ))
            if episode % 10 == 0:
                plt.plot(self.average_rewards)
                plt.plot(self.rewards)
                plt.title("DQN Replay Training Performance Curve")
                plt.xlabel("Episode")
                plt.ylabel("Rewards")
                plt.show()
                
                
        env.close()
        plt.savefig("DQN_Replay_Training_Performance_Curve")
        
        
        
       
        
            
            
            

    def save(self, name):
        self.model.save(name)
        
    def update_counter(self):
        self.counter += 1
        step_size = 5
        self.counter = self.counter % step_size

    def test_trained_model(self, trained_model, num_episodes):
        for episode in range(num_episodes):
            steps = 1000
            trained_state = env.reset()
            ep_reward = 0
            observation_space_dim = env.observation_space.shape[0]
            trained_state = np.reshape(trained_state, [1, observation_space_dim])
            for step in range(steps):
                env.render()
                #time.sleep(0.0003)
                trained_action = np.argmax(trained_model.predict(trained_state)[0])
                next_state, reward, done, info = env.step(trained_action)
                next_state = np.reshape(next_state, [1, observation_space_dim])
                trained_state = next_state
                ep_reward += reward
                if done:
                    break
            
            average_reward_trained = np.mean(self.trained_rewards[-100:])
            self.trained_rewards.append(ep_reward)
            self.average_rewards_trained.append(average_reward_trained)
            
            print("Tr Ep: {}, Ep Reward: {:.2f}, Last Reward: {:.2f}, Avg Reward: {:.2f}".format(episode,
                                                                                             ep_reward, 
                                                                                             reward, 
                                                                                             average_reward_trained))
        env.close()
        plt.plot(self.average_rewards_trained)
        plt.plot(self.trained_rewards)
        plt.title("DQN Replay Trained Performance Curve")
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.show()

            

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env.seed(21)
    np.random.seed(21)
    lr = 0.001
    eps = 1.0
    eps_decay = 0.995
    gamma = 0.99
    training_episodes = 200
    model = DQN(env, lr, gamma, eps, eps_decay)
    model.train(training_episodes)
    # training_episodes = 100
    while (np.mean(model.rewards[-10:]) < 180):
        model.train(training_episodes)
    model.save(name)
    model.save("Dropout_replay_DQN_trained_model3.h5")
    # Dropout_replay_DQN_trained_model3.h5
    trained_model = load_model("replay_DQN_trained_model3.h5")
    model.test_trained_model(trained_model, num_episodes=30)







        
