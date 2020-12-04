import gym
import numpy as np
import time
import wandb

from DQN import DQNAgent
from QLearning import QAgent
from keras.models import load_model



if __name__ == '__main__':
    
    env = gym.make('LunarLander-v2')
    
    # Initialize seeds for reproducability.
    env.seed(21)
    np.random.seed(21)
    
    # Hyperparameters
    learning_rate = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    gamma = 0.99
    deep_layers = 1
    layer_size = 256
    input_layer_mult = 2
    memory_size = 100_000
    batch_size = 64
    
    
    training_episodes = 200
    testing_episodes = 200
    frames = 1000
    
    name = "{}_Deep_Layers_{}_Layer_size_{}_Input_layer_mult_{}_Learning_rate_{}_Gamma_{}_Epsilon_{}_Epsilon_decay_{}_Epsilon_min_{}_Batch_size_{}_Memory_size_Timestamp_{}".format(deep_layers, layer_size, input_layer_mult, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, batch_size, memory_size, int(time.time()))
    
    # For Weights and Biases parameter Sweeps
    run = wandb.init(project=name,
                              config={
                                  "deep_layers": deep_layers,
                                  "layer_size": layer_size,
                                  "input_layer_mult": input_layer_mult,
                                  "learning_rate": learning_rate,
                                  "gamma": gamma,
                                  "epsilon": epsilon,
                                  "epsilon_decay": epsilon_decay,
                                  "epsilon_min": epsilon_min,
                                  "batch_size": batch_size,
                                  "memory_size": memory_size
                            })
        
    # Utilize the hyperparameters of the model like this: config.parameter
    config = wandb.config
    
    model = DQNAgent(env, config, training_episodes, testing_episodes, frames)

    # for deep_layers in deep_layers:
    #     model = DQNAgent(env, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, deep_layers, layer_size, 
    #                 input_layer_mult, memory_size, batch_size, training_episodes, testing_episodes, frames)
    #     model.train()
    
    # while (np.mean(model.rewards[-10:]) < 180):
    #     model.train()
    # model.save("replay_DQN_trained_model5.h5")
    # trained_model = load_model("replay_DQN_trained_model3.h5")
    # model.test_trained_model(trained_model, num_episodes=30)
    
    
    # QLearning Cartpole:
    # env = gym.make('CartPole-v0')
    # buckets = (3, 3, 6, 6,)
    # min_learning_rate = 0.1
    # divisor = 20
    # buckets = (3,3,6,6,)
    # model = QAgent(env, gamma, min_learning_rate, epsilon_min, divisor, buckets, training_episodes, testing_episodes, frames)
    # model.run()