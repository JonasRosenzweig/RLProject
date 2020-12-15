import math
import wandb
from Agent import Agent
import numpy as np

class QAgent(Agent):
    def __init__(self, env, config):
        Agent.__init__(env, config)
        self.env = env
        self.actionSpace = env.action_space
        self.observationSpace = env.observation_space
        self.name = "QAgent"
        self.config = config
        self.actionSpaceSize = self.env.action_space.n
        self.obsSpaceSize = self.env.observation_space.shape[0]
        self.Qtable = np.zeros(self.config.buckets + (self.actionSpaceSize,))
            
    def discretize(self, state):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
        discretized_state = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(state))]
        discretized_state = [min(self.buckets[i] - 1, max(0, discretized_state[i])) for i in range(len(state))]
        return tuple(discretized_state)
    
    def Act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.actionSpaceSize)
        
        