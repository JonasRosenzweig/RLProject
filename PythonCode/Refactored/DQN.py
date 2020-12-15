import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
from Agent import Agent
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.losses import mean_squared_error
import keras as keras

class DQNAgent(Agent):
    def __init__(self, env, config):
        Agent.__init__(env, config)
        self.env = env
        self.actionSpace = env.action_space
        self.observationSpace = env.observation_space
        self.name = "DQNAgent"
        self.config = config
        self.actionSpaceSize = self.env.action_space.n
        self.obsSpaceSize = self.env.observation_space.shape[0]
        self.memory = deque(maxlen=self.config.memory)
        self.replayCount = 0
        self.trainingFrameCount = 0
        
        self.model = self.createModel()
    def createModel(self):
        model = Sequential()
        model.add(Dense(self.config.layerSize*self.config.layerSizeMult, 
                  input_dim = self.obsSpaceSize, activation = relu))
        for i in range(self.config.deepLayers):
            model.add(Dense(self.config.layerSize, activation = relu))
        model.add(Dense(self.actionSpaceSize, activation = linear))
        model.compile(loss=mean_squared_error,
                      optimizer=Adam(lr=self.config.learningRate))
        print(model.summary())
        return model
    def Act(self, state):
        if np.random.rand() < self.config.epsilon:
            return random.randrange(self.actionSpaceSize)
        bestActions = self.model.predict(state)
        return np.argmax(bestActions[0])
    def addToMem(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def sampleFromMem(self):
        sample = random.sample(self.memory, self.config.batchSize)
        return sample
    def attributesFromSample(self, sample):
        states = np.array([i[0] for i in sample])
        actions = np.array([i[1] for i in sample])
        rewards = np.array([i[2] for i in sample])
        next_states = np.array([i[3] for i in sample])
        done_list = np.array([i[4] for i in sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, done_list
    def learnFromMem(self):
        if len(self.memory) < self.config.batch_size or self.replay_counter != 0:
            return
        if np.mean(self.training_episode_rewards[-10:]) > 100:
            return
        sample = self.sample_from_memory()
        states, actions, rewards, next_states, done_list = self.extract_from_sample(sample)
        targets = rewards + self.config.gamma * (np.amax(self.model.predict_on_batch(next_states), 
                                                         axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.config.batch_size)])
        target_vec[[indexes], [actions]] = targets
        self.model.fit(states, target_vec, epochs=1, verbose=0)
    def save(self, name):
        self.model.save(name)
    def updateCounter(self):
        self.replayCount += 1
        replayStepSize = self.config.replayStepSize
        self.replayCount = self.replayCount % replayStepSize
    
        
        

    
    
        
    
        
    
        
        
        

        
        
        
        