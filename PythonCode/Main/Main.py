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
    learning_rates = [0.0001, 0.001]
    epsilon = 1.0
    epsilon_decay = 0.95
    epsilon_min = 0.01
    gamma = 0.99
    deep_layers = 1
    layer_sizes = [32, 256, 1024]
    input_layer_mult = 2
    memory_sizes = [50_000, 100_000, 1_000_000]
    batch_sizes = [64, 128]
    
    
    training_episodes = 2000
    testing_episodes = 0
    frames = 1000
    hyper_param_counter = 0
    total_runs = len(learning_rates)*len(layer_sizes)*len(memory_sizes)*len(batch_sizes)
    
    name = "DQNAgent"
    for learning_rate in learning_rates:
            for layer_size in layer_sizes:
                for memory_size in memory_sizes:
                    for batch_size in batch_sizes:
                        
                        hyper_param_counter += 1
        
                        name = "WithConfig_Timestamp_{}".format(int(time.time()))
                        # name = "LR_{}_LS_{}_BS_{}_MS_{}_Timestamp_{}".format(int(time.time()))
                        
                        # Hyperparameter setup for final experiment:
                        # Learning rate     Layer Size  Batch Size  Memory Size
                        # 0.001             256         64          100.000
                        # 0.001             256         64          50.000
                        # 0.0001            1024        128         50.000
                        # 0.0001            256         64          100.000
                        # 0.0001            1024        64          100.000
                        
                        
                        # For Weights and Biases parameter Sweeps
                        run = wandb.init(project="DQN-LunarLander-v2_with_Config_GPU",
                                                  config={
                                                      "deep_layers": deep_layers,
                                                      "layer_size": layer_size,
                                                      "input_layer_mult": input_layer_mult,
                                                      "learning_rate": learning_rate,
                                                      "gamma": gamma,
                                                      "epsilon_decay": epsilon_decay,
                                                      "epsilon_min": epsilon_min,
                                                      "batch_size": batch_size,
                                                      "memory_size": memory_size,
                                                      "name": name
                                                }, name = name, allow_val_change = True)
                            
                        # Utilize the hyperparameters of the model like this: config.parameter
                        config = wandb.config
                       
                        
                        model = DQNAgent(env, config, epsilon, training_episodes, testing_episodes, frames)
                        
                        model.train()
                        print("Run {} of {}.".format(hyper_param_counter, total_runs))
                        model_dir = "saved_models"
                        model_save_name = model_dir + "DQNModel_{}_".format(int(time.time())) + "sb.h5"
                        model.save(model_save_name)
                        #model.test_trained_model(load_model("DQNAgentModel100Rewardsave.h5"))
            
    # for deep_layers in deep_layers:
    #     for learning_rate in learning_rates:
        
    #         model = DQNAgent(env, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, deep_layers, layer_size, 
    #                     input_layer_mult, memory_size, batch_size, training_episodes, testing_episodes, frames)
    #         model.train()
    
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