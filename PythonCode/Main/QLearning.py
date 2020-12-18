

from RLAgent import RLAgent

# Import wandb for logging the agents' runs.
import wandb

import gym

import numpy as np
import matplotlib.pyplot as plt
import math

class QAgent(RLAgent):
    """
    A class used to build Q-learning algorithms.
    
    This subclass of RLAgent allows to build Q-learning algorithms with
    externally specified (hyper-)parameters and provides the necessary methods
    for training an algorithm.
    
    Attributes
    ----------
    env : env
        The environment of a RL algorithm.
        
    gamma : float
        The discount factor for future rewards.
        
    learning_rate : float
        The learning rate.
        
    epsilon : float
        The epsilon.
        
    epsilon_min : float
        The minimum epsilon value.
        
    epsilon_decay : float
        The epsilon decay.
        
    divisor : int or float
        Number used in learning rate decay.
        
    buckets : tuple of ints
        Tuple used to discretize the observation space.
        
    training_episodes : int
        Maximum number of training episodes.
        
    testing_episodes : int
        Maximum number of testing episodes.
        
    frames : int
        Maximum number of frames during an episode.
        
    Methods
    -------
    discretize(state)
        Takes an observation space and returns a discreticed observation space.
    
    get_action(state, epsilon)
        Chooses and returns an action from the action space.
        
    update_Q(state, action, reward, next_state, learning_rate)
        Updates the state-action pairs and Q values in the Q-table.
    
    run()
        Loop to train the Q-agent.
    """
    
    def __init__(self, env, gamma, learning_rate, epsilon, epsilon_min, divisor,
                 buckets, training_episodes, testing_episodes, frames):
        
        RLAgent.__init__(self, env, training_episodes, testing_episodes, frames)
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.divisor = divisor 
        self.buckets = (3,3,6,6,)
        
        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))
        
    def discretize(self, state):
        upper_bounds = [self.env.observation_space.high[0], 0.5,
                        self.env.observation_space.high[2], math.radians(50)]
        
        lower_bounds = [self.env.observation_space.low[0], -0.5,
                        self.env.observation_space.low[2], -math.radians(50)]
        
        ratios = [(state[i] + abs(lower_bounds[i])) /
                  (upper_bounds[i] - lower_bounds[i])
                  for i in range(len(state))]
        
        discretized_state = [int(round((self.buckets[i] - 1) *ratios[i]))
                             for i in range(len(state))]
        
        discretized_state = [min(self.buckets[i] -1,
                                 max(0, discretized_state[i]))
                             for i in range(len(state))]
        
        return tuple(discretized_state)
    
    def get_action(self, state, epsilon):
        return self.env.action_space.sample() if (
            np.random.random() <= epsilon) else np.argmax(self.Q[state])
    
    def update_Q(self, state, action, reward, next_state, learning_rate):
        self.Q[state][action] += learning_rate * (
                reward + self.gamma * np.max(
                self.Q[next_state]) - self.Q[state][action])
        
        return max(self.epsilon_min, min(1.0,
                     1.0 - math.log10((episode + 1)/self.divisor)))
    
    def run(self):
        print("running")
        for episode in range(self.training_episodes):
            
            episode_reward = 0
            discretized_state = self.discretize(self.env.reset())
            done = False
            
            while not done:
                
                action = self.get_action(discretized_state, self.epsilon)
                state, reward, done, info = self.env.step(action)
                next_state = self.discretize(state)
                self.update_Q(discretized_state, action, reward, next_state,
                              self.learning_rate)
                discretized_state = next_state
                # env.render()
                episode_reward += reward
            average_reward = np.mean(self.training_episode_rewards)
            
            if average_reward > 200:
                break
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            self.training_episode_rewards.append(episode_reward)
            self.training_average_rewards.append(average_reward)
            print("Episode: {}, total_reward: {:.2f}, Epsilon: {:.2f}".format(
                episode, episode_reward, self.epsilon))
            
            
        self.env.close()
        
