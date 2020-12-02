import gym
import numpy as np


from DQN import DQNAgent
from QLearning import QAgent
from keras.models import load_model



if __name__ == '__main__':
    
    env = gym.make('LunarLander-v2')
    env.seed(21)
    np.random.seed(21)
    learning_rate = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    gamma = 0.99
    training_episodes = 200
    testing_episodes = 200
    frames = 1000
    num_deep_layers = [1, 2, 3]
    num_neurons = 256
    input_layer_mult = 2
    memory_size = 100_000
    batch_size = 64

    for deep_layers in num_deep_layers:
        model = DQNAgent(env, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, deep_layers, neurons, 
                    input_layer_mult, memory_size, batch_size, training_episodes, testing_episodes, frames)
    model.train()
    
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