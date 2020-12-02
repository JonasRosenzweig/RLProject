import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import wandb

from RLAgent import RLAgent

from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.losses import mean_squared_error




class DQNAgent(RLAgent):
    def __init__(self, env, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, 
                 deep_layers, neurons, input_layer_mult, memory_size, batch_size, training_episodes, testing_episodes, frames):
        RLAgent.__init__(self, env, training_episodes, testing_episodes, frames)
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.action_space_dim = self.env.action_space.n
        self.observation_space_dim = self.env.observation_space.shape[0]
        
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.batch_size = batch_size
        self.replay_counter = 0
        self.training_frame_count = 0
        
        self.deep_layers = deep_layers
        self.neurons = neurons
        self.input_layer_mult = input_layer_mult
        
        
        self.model = self.initialize_model()
        
    def initialize_model(self):
        model = Sequential()
        model.add(Dense(self.neurons*self.input_layer_mult, input_dim = self.observation_space_dim, activation=relu))
        for i in range(self.deep_layers):
            model.add(Dense(self.neurons, activation=relu))
        model.add(Dense(self.action_space_dim, activation=linear))
        model.compile(loss=mean_squared_error, optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space_dim)
        predicted_actions = self.model.predict(state)
        return np.argmax(predicted_actions[0])
    
    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample_from_memory(self):
        sample = random.sample(self.memory, self.batch_size)
        return sample
    
    def extract_from_sample(self, sample):
        states = np.array([i[0] for i in sample])
        actions = np.array([i[1] for i in sample])
        rewards = np.array([i[2] for i in sample])
        next_states = np.array([i[3] for i in sample])
        done_list = np.array([i[4] for i in sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, done_list
    
    def learn_from_memory(self):
        if len(self.memory) < self.batch_size or self.replay_counter != 0:
            return
        if np.mean(self.training_episode_rewards[-10:]) > 180:
            return
        
        sample = self.sample_from_memory()
        states, actions, rewards, next_states, done_list = self.extract_from_sample(sample)
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets
        
        self.model.fit(states, target_vec, epochs=1, verbose=0)
        
    def train(self):
        wandb.init(project="LanderDQN", name="Performance")
        for episode in range(self.training_episodes):
            steps = self.frames
            state = self.env.reset()
            episode_reward = 0
            state = np.reshape(state, [1, self.observation_space_dim])
            episode_frame_count = 0
            for step in range(steps):
                #env.render()
                exploit_action = self.get_action(state)
                next_state, reward, done, info = self.env.step(exploit_action)
                next_state = np.reshape(next_state, [1, self.observation_space_dim])
                episode_reward += reward
                self.training_frame_count += 1
                episode_frame_count += 1
                self.add_to_memory(state, exploit_action, reward, next_state, done)
                state = next_state
                self.update_counter()
                self.learn_from_memory()
                if done: 
                    break
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            average_reward = np.mean(self.training_episode_rewards[-100:])
            if average_reward > 200:
                break
            self.training_episode_rewards.append(episode_reward)
            self.training_average_rewards.append(average_reward)
            wandb.log({'reward': average_reward, 'last reward': reward, 'epsilon': self.epsilon}, step=episode)
            
            print("""Tr Ep: {}, Ep Reward: {:.2f}, Last Reward: {:.2f}, Total Frames: {}, Frames: {}, Avg Reward: {:.2f}, 
                  Eps: {:.2f},""".format(episode, episode_reward, reward, self.training_frame_count, episode_frame_count, average_reward, self.epsilon))
                      
            if episode % 10 == 0:
                plt.plot(self.training_episode_rewards)
                plt.plot(self.training_average_rewards)
                plt.title("DQN Replay Training Performance Curve")
                plt.xlabel("Episode")
                plt.ylabel("Rewards")
                plt.show()
                       
        self.env.close()
        plt.savefig("DQN_Replay_Training_Performance_Curve")
            
    def save(self, name):
        self.model.save(name)
    
    def update_counter(self):
        self.replay_counter += 1
        step_size = 5
        self.replay_counter = self.replay_counter % step_size
    
    def test_trained_model(self, trained_model):
        for episode in range(self.testing_episodes):
            steps = self.frames
            trained_state = self.env.reset()
            episode_reward = 0
            observation_space_dim = self.env.observation_space.shape[0]
            trained_state = np.reshape(trained_state, [1, observation_space_dim])
            for step in range(steps):
                self.env.render()
                trained_action = np.argmax(trained_model.predict(trained_state)[0])
                next_state, reward, done, info = self.env.step(trained_action)
                next_state = np.reshape(next_state, [1, observation_space_dim])
                trained_state = next_state
                episode_reward += reward
                if done:
                    break
            
            average_reward_trained = np.mean(self.trained_rewards[-100:])
            self.test_episode_rewards.append(episode_reward)
            self.test_average_rewards.append(average_reward_trained)
            
            print("Tr Ep: {}, Ep Reward: {:.2f}, Last Reward: {:.2f}, Avg Reward: {:.2f}".format(episode,
                                                                                             episode_reward, 
                                                                                                 reward, 
                                                                                                 average_reward_trained))
        self.env.close()
        plt.plot(self.test_episode_rewards)
        plt.plot(self.test_average_rewards)
        plt.title("DQN Replay Trained Performance Curve")
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.show()
            
            
            
        
        
    
    
    
    
        
        