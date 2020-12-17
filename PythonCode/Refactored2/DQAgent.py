import numpy as np
import random
from collections import deque

from QAgent import QAgent

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.losses import mean_squared_error

class DQAgent(QAgent):
    """
    A class used to build Deep Q-learning neural networks.
    
    This subclass of RLAgent allows to create Deep Q-learning neural networks 
    (DQNs) with externally specified (hyper-)parameters and provides methods 
    for training and testing models.
    
    Attributes
    ----------
    env : env
        The environment of a RL algorithm.
        
    config : config of parameters
        Wandb's config file for storing and tracking parameters.   
        
    epsilon : float
        Used to determine whether a random or exploitative action is taken. 
        
    training_episodes : int
        Maximum number of training episodes. 
        
    testing_episodes : int
        Maximum number of testing episodes. 
        
    frames : int
        Maximum number of frames during an episode.
        
    name : str
        The name of the DQN.
        
    action_space_dim : int
        The number of values in the action space.
        
    observation_space_dim : int
        The number of values in the observation space.
        
    memory : int
        Maximum length of the memory_size.
        
    replay_counter : int
        A counter used to determine when the agent should learn from 
        experience.
        
    training_frame_count : int
        A counter used to keep track of how many frames the agent ran 
        through during training.
        
    model : model
        A model of a DQN.
            
    Methods
    -------
    initialize()
        Constructs and returns a DQN model.
        
    randomAct(state)
        Chooses and returns a random action from the action space.
        
    policyAct(state)
        Chooses and returns an action from the action space based on the
        current policy.
    
    act(state)
        Chooses either randomAct or policyAct an returns an action.
        
    ---> What about the _ naming convention from the older code?
    
    add_to_memory(state, action, reward, next_state, done)
        Adds an experience tuple to the memory list.
        
    sample_from_memory()
        Randomly chooses and returns a memory from the memory list.
        
    extract_from_sample(sample)
        Extracts and returns an experience tuple from a past experience.
    
    learn_from_memory()
        Adjusts the weights of the DQN utilizing a past experience.
        
    train()
        Loop to train the DQN.
        
    save()
        Saves the model.
        
    update_counter()
        Updates the counter variable.
        
    test_trained_model(trained_model)
        Loop to test a trained model.
    """
    
    def __init__(self, env, config):
        QAgent.__init__(self, env, config)
        
        self.name = self.config.name
        self.memory = deque(maxlen=self.config.memory_size)
        self.model = self.initialize()
        
        self.replay_counter = 0
    
    def initialize(self):
        model = Sequential()
        model.add(Dense(self.config.layer_size*self.config.input_layer_mult, 
                        input_dim = 10, activation=relu)) 
        # input_dim = input shape value 
        # batch_size is x in receied input: [x, 1]
        for i in range(self.config.deep_layers):                            
            model.add(Dense(self.config.layer_size, activation=relu))
        model.add(Dense(self.action_space_size, activation=linear))
        model.compile(loss=mean_squared_error, 
                      optimizer=Adam(lr=self.config.learning_rate))
        print(model.summary())
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
        sample = random.sample(self.memory, self.config.batch_size)
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
        self.replay_counter += 1
        self.replay_counter = self.replay_counter % self.config.replay_step_size
    
    def learnFromMemory(self):
        if len(self.memory) < self.config.batch_size or self.replay_counter != 0:
            return
        if np.mean(self.training_episode_rewards[-10:]) > 100:
            return
        sample = self.sampleFromMemory()

        states, actions, rewards, next_states, done_list = self.extractFromSample(sample)
        targets = rewards + self.config.gamma * (np.amax(
                self.model.predict_on_batch(next_states), axis=1)) * (
                1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.config.batch_size)])
        target_vec[[indexes], [actions]] = targets
        self.model.fit(states, target_vec, epochs=1, verbose=0)
        
    def save(self, name):
        self.model.save(name)
        
        
        
        
        
        
        
        
    
        
        
        