import math
import random
import numpy as np

from Agent import Agent



class QAgent(Agent):
    def __init__(self, env, config):
        Agent.__init__(self, env, config)
        
        self.buckets = self.config.buckets
        self.Qtable = np.zeros(tuple(self.config.buckets) + (env.action_space.n,))
        
    def discretize(self, state):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
        discretized_state = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(state))]
        discretized_state = [min(self.buckets[i] - 1, max(0, discretized_state[i])) for i in range(len(state))]
        return tuple(discretized_state)
    
    def policyAct(self, state):
        np.argmax(self.Qtable[state])
    
    def updateQ(self, state, action, reward, next_state):
        self.Qtable[state][action] += self.config.learning_rate * (reward + self.config.gamma * np.max(self.Qtable[next_state]) - self.Qtable[state][action])
        

