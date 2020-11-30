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
from keras import Sequential
from keras import Activation, Flatten, MaxPooling2D, Dropout
from keras.layers import Dense, Conv2D
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.losser import mean_squared_error
from keras.models import load_model

# Using these parameters in nested loops would allow for creating different combinations of NNs and find the best combination
deep_dense_layers = [1, 2, 3]
num_neurons = [32, 64, 128]


class DQN:
    def __init__(self, env, lr, gamma, epsilon, epsilon_decay):
        
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.counter = 0
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rewards_list = []
        
        self.replay_memory_buffer = deque(maxlen = 500_000)
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.num_action_space = self.action_space.n
        self.num_observation_space = env.observation_space.shape[0]
        
        self.model = self.initialize_model()
        
    def initialize_model(self):
        model = Sequential()
        
        # Input layer
        model.add(Dense(512, input_dim = self.num_observation_space, activation = relu))
        model.add(Dropout(0.2))
        
        # Deep layers
        model.add(Dense(256, activation = relu))
        model.add(Dropout(0.1))
        
        # Output layer
        model.add(Dense(self.num_action_space, activation = linear))

