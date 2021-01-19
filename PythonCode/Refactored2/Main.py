import gym
import numpy as np
import time
import wandb

from QAgent import QAgent
from DQAgent import DQAgent
from Run import Run
from Experiments import  gammaExperiment, learningRateExperiment, epsilonDecayExperiment

if __name__ == '__main__':
    
    env = gym.make('LunarLander-v2')
    
    
    env.seed(21)
    np.random.seed(21)
    
    learning_rate = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    gamma = 0.99
    
    gammas = [0.99, 0.9, 0.8, 0.7]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    epsilon_decays = [0.999, 0.995, 0.99, 0.9]
    
    buckets = (3,3,6,6,)
    
    
    deep_layers = 1
    layer_size = 128
    input_layer_mult = 2
    
    memory_size = 100_000
    batch_size = 64
    replay_counter = 5
    
    # For running
    experiment_episodes = 2
    training_episodes = 2000
    testing_episodes = 2
    steps = 1000
    render = False
    early_stop = True
    episode_time_limit = 30
    goal = 100
    min_reward = -300
    
    
    name = "LR_{}_LS_{}_BS_{}_MS_{}_Timestamp_{}".format(learning_rate, layer_size, batch_size, memory_size, int(time.time()))
    # name = agent1.name
    
    run = wandb.init(project="Refactor-Test",
                                  config = {"deep_layers": deep_layers,
                                      "layer_size": layer_size,
                                      "buckets": buckets,
                                      "input_layer_mult": input_layer_mult,
                                      "learning_rate": learning_rate,
                                      "epsilon": epsilon,
                                      "replay_counter": replay_counter,
                                      "gamma": gamma,
                                      "epsilon_decay": epsilon_decay,
                                      "epsilon_min": epsilon_min,
                                      "batch_size": batch_size,
                                      "memory_size": memory_size,
                                      "name": name})
            
    
    config = wandb.config
    
    
    # agent2 = QAgent(env, config)
    agent1 = DQAgent(env, config)
    # Train(agent1, agent1_run_config, goal, min_reward)
    
    
    agent1_run_config = {"experiment_episodes": experiment_episodes, 
                         "training_episodes": training_episodes, 
                         "testing_episodes": testing_episodes, 
                         "steps": steps, 
                         "render": render, 
                         "early_stop": early_stop, 
                         "episode_time_limit": episode_time_limit, 
                         "goal": goal, 
                         "min_reward": min_reward}
    
    Run = Run(agent1, agent1_run_config)
    Run.train()
    
    # gamma_experiment_config = {"experiment_episodes": experiment_episodes, "gammas": gammas}
    # gammaExperiment(agent1, gamma_experiment_config)
    
    # learning_rate_experiment_config = {"experiment_episodes": experiment_episodes, "learning_rates": learning_rates}
    # learningRateExperiment(agent1, learning_rate_experiment_config)
    
    # epsilon_decay_experiment_config = {"experiment_episodes": experiment_episodes, "epsilon_decays": epsilon_decay}
    # epsilonDecayExperiment(agent1, epsilon_decay_experiment_config)
    