# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:46:43 2020

@author: Kata
"""
"""
sources:
    https://pythonprogramming.net/convolutional-neural-network-deep-learning-python-tensorflow-keras/
    https://github.com/fakemonk1/Reinforcement-Learning-Lunar_Lander/blob/master/Lunar_Lander.py
"""


import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pickle
import time

import keras
# Sequential NN model is the most common one.
from keras import Sequential
# Layers used for NNs: Conv2D is usually used for image recognition,
# Dense is commonly used, but may be prone to overfitting.
from keras.layers import Dense, Conv2D
# Allows using functions such as flattening (when trying to change from Conv2D to Dense layer)
# or MaxPooling, which is used in Conv2D layers.
from keras import Activation, Flatten, MaxPooling2D, Dropout
# Activation functions: relu (rectified linear) is standard in NN
# linear is used for the final layer to get just one possible answer.
from keras.activations import relu, linear
# Standard optimizer is adam.
from keras.optimizers import Adam
from keras.losser import mean_squared_error
from keras.models import load_model

# Using these parameters in nested loops would allow for creating different combinations of NNs and find the best combination
deep_dense_layers = [1, 2, 3]
num_neurons = [32, 64, 128]


class DQN:
    def __init__(self, env, lr, gamma, epsilon, epsilon_decay):
        
        # Initializes variables based on the environment (e.g. LunarLander-V2, Cartpole-V0, etc.)
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_action_space = self.action_space.n
        self.num_observation_space = env.observation_space.shape[0]
        
        # Initializes variables based on the hyperparameters given.
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rewards_list = []
        
        # Initializes variables the same every time, as given below.
        self.replay_memory_buffer = deque(maxlen = 500_000)
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.counter = 0
        
        self.model = self.initialize_model()
        
    # Constructs model using sequential model and different (deep) layers.
    def initialize_model(self):
        model = Sequential()
        
        
        # Input layer (based on observation space of the environment)
        model.add(Dense(512, input_dim = self.num_observation_space, activation = relu))
        model.add(Dropout(0.2))
        
        # Deep layers
        model.add(Dense(256, activation = relu))
        model.add(Dropout(0.1))
        
        # Output layer (based on action space of the environment)
        model.add(Dense(self.num_action_space, activation = linear))
        
        
        # Compile the model giving the loss and the optimizer as an argument.
        model.compile(loss = mean_squared_error, optimizer = Adam (lr = self.lr))
        
        # Prints out the stats of the model to give an overview over what was just created.
        print(model.summary())
        
        
        return model

    # Decide whether to take an exploratory or exploitative action.
    def get_action(self, state):
        
        # Based on a random number 0 <= n <= 1, if n smaller than the current epsilon e, select random action based on the action space of the environment.
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_action_space)

        # Otherwise let the model decide the best action in the current environment state based on the momentary policy.
        predicted_actions = self.model.predict(state)
        
        # Return the action to be taken in the current state.
        return np.argmax(predicted_actions[0])
    
    def learn_and_update_weights_by_reply(self):
        
        # replay_memory_buffer size check (Needs rewording / more understanding)
        if len(self.replay_memory_buffer) < self.batch_size or self.counter != 0:
            return
        
        # If the model has been completing the task with a desirable reward for a while, stop it to prevent it from overfitting.
        if np.mean(self.rewards_list[-10]) > 180:
            return
        
        # Choose a random past experience from the replay memory
        random_sample = self.get_randomo_sample_from_replay_mem()
        # Convert the chosen experience's attributes to the needed parameters (state, action, etc.)
        states, actions, rewards, next_states, done_list = self.get_attributes_from_sample(random_sample)
        
        # Update the rewards based on the discount factor (gamma) influencing the next set of states. (Needs revising / more understanding)
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis = 1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets
        
        # Adjusts the policy based on states, target vectors and other things (needs more understanding)
        self.model.fit(states, target_vec, epochs = 1, verbose = 0)
        
    def get_attributes_from_sample(self, random_sample):
        
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
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        