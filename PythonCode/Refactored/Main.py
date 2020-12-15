import gym
import numpy as np
import time
import wandb

from DQN import DQNAgent
from keras.models import load_model
from RunInterface import trainDQN

if __name__ == '__main__':
    
    env = gym.make('LunarLander-v2')
    env.seed(21)
    np.random.seed(21)
    
    epsilon = 1.0
    epsilonDecay = 0.995
    epsilonMin = 0.01
    gamma = 0.99
    learningRate = 0.001
    
    layerSizeMult = 2
    layerSize = 128
    deepLayers = 1
    
    batchSize = 64
    memory = 100_000
    replayStepSize = 5
    
    trainingEpisodes = 2
    testingEpisodes = 2
    frames = 1000
    
    name = "DQN"
    
    
    
    
    run = wandb.init(project="test",
                                  config={
                                      "trainingEpisodes": trainingEpisodes,
                                      "testingEpisodes": testingEpisodes,
                                      "frames": frames,
                                      "epsilon": epsilon,
                                      "deepLayers": deepLayers,
                                      "layerSize": layerSize, 
                                      "layerSizeMult": layerSizeMult,
                                      "learningRate": learningRate,
                                      "gamma": gamma,
                                      "epsilonDecay": epsilonDecay,
                                      "epsilonMin": epsilonMin,
                                      "batchSize": batchSize,
                                      "memory": memory,
                                      "name":name,
                                      "replay_step_size" : replayStepSize,
                                      
                                }, name = name, allow_val_change = True)
    config = wandb.config
    
    agent = DQNAgent(env, config)
    trainDQN(DQNAgent)
    
    
    
    