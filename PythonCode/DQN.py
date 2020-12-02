import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import wandb
import time
import RLAgent

from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import load_model

import pickle
from matplotlib import pyplot as plt

class DQN(RLAgent):
    def __init__(self, learning_rate, gamma, epsilon, epsilon_decay):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilonm = epsilon
        self.epsilon_decay = epsilon_decay
        
        self.action_space_dim = self.action_space.n
        self.observation_space_dim = self.observation_space.shape[0]
        