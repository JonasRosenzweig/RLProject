# -*- coding: utf-8 -*-


# Import the gym environment from OpenAI
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
# Allows for logging the results online via "Weights and Biases"
# source: https://colab.research.google.com/drive/1aEv8Haa3ppfClcCiC2TB8WLHB4jnY_Ds
import wandb
import keras as keras
from RLAgent import RLAgent

# Sequential NN model is the most common one.
from keras import Sequential
# Layers used for NNs: Conv2D is usually used for image recognition,
# Dense is commonly used, but may be prone to overfitting.
from keras.layers import Dense
# Allows using functions such as Flatten* (when trying to change from Conv2D to Dense layer)
# or MaxPooling, which is used in Conv2D layers.
# * Flatten converts 3D feature maps (Conv2D) into 1D feature vectors
# from keras import Flatten, MaxPooling2D, Dropout
# Activation functions: relu (rectified linear) is standard in NN
# linear is used for the final layer to get just one possible answer.
from keras.activations import relu, linear
# Standard optimizer is adam.
from keras.optimizers import Adam
from keras.losses import mean_squared_error
import time 
from wandb.integration.keras import WandbCallback

# metrics = [
#       keras.metrics.BinaryAccuracy(name='accuracy'),
#       keras.metrics.Precision(name='precision'),
#       keras.metrics.Recall(name='recall'),
# ]



class DQNAgent(RLAgent):
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
        A counter used to determine when the agent should learn from experience.        
    training_frame_count : int
        A counter used to keep track of how many frames the agent ran 
        through during training.    
    model : model
        A model of a DQN.
            
    Methods (Do not mention parameter "self")
    -------
    initialize_model()
        Constructs and returns a DQN model.
        
    get_action(state)
        Chooses and returns an action from the action space.
        
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
        
        
    Returns (Expected returned values)
    -------
    
    See Also (Referring to related code)
    --------
    
    Notes (Referring to additional information)
    -----
    
    References (Citing the sources in Notes)
    ----------
    """
    
    def __init__(self, env, config, epsilon, training_episodes, testing_episodes, frames):
        """
        Parameters
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
        
        Returns
        -------
        None.

        """
        RLAgent.__init__(self, env, training_episodes, testing_episodes, frames)
        
        self.epsilon = epsilon
        self.name = config.name
        
        self.action_space_dim = self.env.action_space.n
        self.observation_space_dim = self.env.observation_space.shape[0]
        
        # Config has all hyperparameters stored.
        self.config = config
        
        self.memory = deque(maxlen=self.config.memory_size)
        self.replay_counter = 0
        
        # Keep track of how many frames the model ran through in total.
        self.training_frame_count = 0
        

        
        

        self.model = self.initialize_model()
        
    # Constructs model using sequential model and different (deep) layers.
    def initialize_model(self):
        model = Sequential()
        
        # Input layer (based on observation space of the environment)
        model.add(Dense(self.config.layer_size*self.config.input_layer_mult, input_dim = self.observation_space_dim, activation=relu))
        
        # Deep layers
        for i in range(self.config.deep_layers):
            model.add(Dense(self.config.layer_size, activation=relu))
        
        # Output layer (based on action space of the environment)
        model.add(Dense(self.action_space_dim, activation=linear))
        
        # Compile the model giving the loss and the optimizer as an argument.
        model.compile(loss=mean_squared_error, optimizer=Adam(lr=self.config.learning_rate))
        
        # Prints out the stats of the model to give an overview over what was just created.
        #print(self.config.name)
        
        
        print(model.summary())
        
        return model
 
    # Decide whether to take an exploratory or exploitative action.
    def get_action(self, state):
        
        # Based on a random number 0 <= n <= 1, if n smaller than the current epsilon e, select random action based on the action space of the environment.
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space_dim)
        
        # Otherwise let the model decide the best action in the current environment state based on the momentary policy.
        predicted_actions = self.model.predict(state)
        
        # Return the action to be taken in the current state.
        return np.argmax(predicted_actions[0])
    
    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    # Choose a random past experience from the replay memory
    def sample_from_memory(self):
        sample = random.sample(self.memory, self.config.batch_size)
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
        
        # replay_memory_buffer size check (Needs rewording / more understanding)
        if len(self.memory) < self.config.batch_size or self.replay_counter != 0:
            return
        
        # If the model has been completing the task with a desirable reward for a while, stop it to prevent it from overfitting.
        if np.mean(self.training_episode_rewards[-10:]) > 100:
            return
        
        sample = self.sample_from_memory()
        
        # Convert the chosen experience's attributes to the needed parameters (state, action, etc.)
        states, actions, rewards, next_states, done_list = self.extract_from_sample(sample)
        targets = rewards + self.config.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.config.batch_size)])
        target_vec[[indexes], [actions]] = targets
        
        # Adjusts the policy based on states, target vectors and other things (needs more understanding)
        self.model.fit(states, target_vec, epochs=1, verbose=0)
        # for callbacks : , callbacks=[WandbCallback(data_type="image")] problem with wandb logging, though needed for sweeps
        
        
    def train(self):
        start_time = time.time()
        
        #wandb.init(project="DQN-LunarLander-v2_with_Config", name=self.config.name)
        
        for episode in range(self.training_episodes):
            
            steps = self.frames
            state = self.env.reset()
            episode_reward = 0
            state = np.reshape(state, [1, self.observation_space_dim])
            episode_frame_count = 0
            
            for step in range(steps):
                
                #self.env.render()
                
                # Decide what action to take.
                exploit_action = self.get_action(state)
                
                # Step to next environment state with current environment state tuple.
                next_state, reward, done, info = self.env.step(exploit_action)
                
                # Update the next state based on the tuple and the observation space.
                next_state = np.reshape(next_state, [1, self.observation_space_dim])
                
                # Add the reward for this step to the episode reward
                episode_reward += reward
                self.training_frame_count += 1
                episode_frame_count += 1
                
                # Add the experience of the state-action pair to the replay memory
                self.add_to_memory(state, exploit_action, reward, next_state, done)
                
                # Progress to the next state by changing the current state to become the next state.
                state = next_state
                
                # Update counter used in the replay memory buffer size check.
                self.update_counter()
                
                # Update the weight connections within the layers.
                self.learn_from_memory()
                
                if done: 
                    break
            
            # Reduce the epsilon based on decay rate to move the focus of the NN from exploration to exploitation over time. 
            if self.epsilon > self.config.epsilon_min:
                self.epsilon *= self.config.epsilon_decay
            
            average_reward = np.mean(self.training_episode_rewards[-100:])
            # Stop if the model has solved the environment (reward must average above 200).
            if average_reward > 200:
                break
            if average_reward < -400 and episode > 100:
                break
            if average_reward < -300 and episode > 200:
                break
            if average_reward < -200 and episode > 300:
                break
            train_time_minutes = (time.time() - start_time)/60
            if train_time_minutes > 30:
                break
            
            
            # Add the episode reward to the list of episodes_rewards for the episodes    
            self.training_episode_rewards.append(episode_reward)
            self.training_average_rewards.append(average_reward)
            wandb.log({'average reward': average_reward, 'last reward': reward, 'epsilon': self.epsilon, 'episode': episode }, step=episode)
            
            # Print out the episode's results with additional information.
            print("""Episode: {}\t\t\t|| Episode Reward: {:.2f}
Last Frame Reward: {:.2f}\t|| Average Reward: {:.2f}\t|| Epsilon: {:.2f}
Frames this episode: {}\t\t|| Total Frames trained: {}\n"""
                .format(episode, episode_reward, reward, average_reward, self.epsilon, episode_frame_count, self.training_frame_count))
            # print("""Tr Ep: {}, Ep Reward: {:.2f}, Last Reward: {:.2f}, Total Frames: {}, Frames: {}, Avg Reward: {:.2f}, 
                   # Eps: {:.2f},""".format(episode, episode_reward, reward, self.training_frame_count, episode_frame_count, average_reward, self.epsilon))
                      
            # if episode % 50 == 0:
            #     plt.plot(self.training_episode_rewards)
            #     plt.plot(self.training_average_rewards)
            #     plt.title("DQN Replay Training Performance Curve")
            #     plt.xlabel("Episode")
            #     plt.ylabel("Rewards")
            #     plt.show()
        
        self.env.close()
        
        # figname = "Figure_"
        # figname = self.config.name
        # plt.savefig("DQN_Replay_Training_Performance_Curve")
            
    def save(self, name):
        self.model.save(name)
    
    def update_counter(self):
        self.replay_counter += 1
        step_size = 5
        self.replay_counter = self.replay_counter % step_size
    
    # Makes a validation run of a trained model, which is very similar to a training run.
    def test_trained_model(self, trained_model):
        
        for episode in range(self.testing_episodes):
        
            steps = self.frames
            trained_state = self.env.reset()
            episode_reward = 0
            observation_space_dim = self.env.observation_space.shape[0]
            trained_state = np.reshape(trained_state, [1, observation_space_dim])
            
            for step in range(steps):
            
                # self.env.render()
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
            
            
            print("""Episode: {}\t\t\t|| Episode Reward: {:.2f}\
Last Frame Reward: {:.2f}\t|| Average Reward: {:.2f}"""
              .format(episode, episode_reward, reward, average_reward_trained))
                
            # print("Tr Ep: {}, Ep Reward: {:.2f}, Last Reward: {:.2f}, Avg Reward: {:.2f}".format(episode,
            #                                                                                  episode_reward, 
            #                                                                                      reward, 
            #                                                                                      average_reward_trained))
        
        self.env.close()
        # plt.plot(self.test_episode_rewards)
        # plt.plot(self.test_average_rewards)
        # plt.title("DQN Replay Trained Performance Curve")
        # plt.xlabel("Episode")
        # plt.ylabel("Rewards")
        # plt.show()
            
            
            
        
        
    
    
    
    
        
        