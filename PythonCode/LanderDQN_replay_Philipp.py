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
        
        # Input layer
        model.add(Dense(512, input_dim = self.num_observation_space, activation = relu))
        model.add(Dropout(0.2))
        
        # Deep layers
        model.add(Dense(256, activation = relu))
        model.add(Dropout(0.1))
        
        # Output layer
        model.add(Dense(self.num_action_space, activation = linear))
        
        

