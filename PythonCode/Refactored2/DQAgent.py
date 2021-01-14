import numpy as np
import random
from collections import deque

from QAgent import QAgent

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input
# from keras import Sequential
# from keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error

class DQAgent(QAgent):
    def __init__(self, env, config):
        QAgent.__init__(self, env, config)
        
        self.memory = deque(maxlen=self.config.memory_size)
        self.model = self.initialize()
    
    def initialize(self):
        
        inputs = Input(shape=(8,))
        
        dense = Dense(self.config.layer_size * self.config.input_layer_mult, activation = relu)
        x = dense(inputs)
        x = Dense(self.config.layer_size, activation = relu)(x)
        
        outputs = layers.Dense(self.action_space_size, activation = linear)(x)
        
        model = keras.Model(inputs = inputs, outputs = outputs, name = self.name)

        model.compile(loss = mean_squared_error, optimizer = Adam(lr = self.config.learning_rate))
        model.summary()

        return model
    
    def randomAct(self, state):
        return random.randrange(self.action_space_size)
    
    def policyAct(self, state):
        predicted_actions = self.model.predict(state)
        return np.argmax(predicted_actions[0])
    
    def act(self, state):
        if np.random.random() <= self.config['epsilon']:
            return self.randomAct(state)
        else:
            return self.policyAct(state)
    
    def addToMemory(self, state, action, reward, next_state, done):
        self.memory.append((self, state, action, reward, next_state, done))
        
    def sampleFromMemory(self):
        sample = np.random.sample(self.memory, self.config.batch_size)
        return sample
    
    def extractFromSample(self, sample):
        states = np.array([i[0] for i in sample])
        actions = np.array([i[1] for i in sample])
        rewards = np.array([i[2] for i in sample])
        next_states = np.array([i[3] for i in sample])
        done_list = np.array([i[4] for i in sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, done_list
    
    def updateReplayCount(self):
        self.config.replay_counter += 1
        self.config.replay_counter = self.replay_counter % self.config.replay_step_size
    
    def learnFromMemory(self):
        if len(self.memory) < self.config.batch_size or self.config.replay_counter != 0:
            return
        if np.mean(self.training_episode_rewards[-10:]) > 100:
            return
        sample = self.sampleFromMemory()

        states, actions, rewards, next_states, done_list = self.extractFromSample(sample)
        targets = rewards + self.config.gamma * (np.amax(self.model.predict_on_batch(next_states), 
                                                         axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.config.batch_size)])
        target_vec[[indexes], [actions]] = targets
        self.model.fit(states, target_vec, epochs=1, verbose=0)
        
    def save(self, name):
        self.model.save(name)
        
        
        
        
        
        
        
    
        
        
        