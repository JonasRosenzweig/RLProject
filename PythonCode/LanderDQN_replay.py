import gym
import random
import numpy as np
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import load_model

from collections import deque


class DQN:
    def __init__(self, env, lr, eps, eps_decay):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.action_space_dim = self.action_space.n
        self.observation_space_dim = self.observation_space.shape[0]
        self.lr = lr
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = 0.01
        self.rewards = []
        self.training_frame_count = 0
        self.model = self.initialize_model()
        self.replay_memory_buffer = deque(maxlen=500000)
        self.batch_size = 64
        self.counter_for_replay = 0

    def initialize_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim = self.observation_space_dim, activation=relu))
        model.add(Dense(16, activation=relu))
        model.add(Dense(self.action_space_dim, activation=linear))
        model.compile(loss=mean_squared_error, optimizer=Adam(lr=self.lr))
        print(model.summary())
        return model

    def get_action(self, state):
        if np.random.rand() < self.eps:
            return random.randrange(self.action_space_dim)
        predicted_actions = self.model.predict(state)
        return np.argmax(predicted_actions[0])

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    def learn_and_update_weights_by_reply(self):

        # replay_memory_buffer size check
        if len(self.replay_memory_buffer) < self.batch_size or self.counter != 0:
            return

        # Early Stopping
        if np.mean(self.rewards_list[-10:]) > 180:
            return

        random_sample = self.get_random_sample_from_replay_mem()
        states, actions, rewards, next_states, done_list = self.get_attribues_from_sample(random_sample)
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets

        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def get_attribues_from_sample(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, done_list

    def get_random_sample_from_replay_mem(self):
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        return random_sample

    def update_counter(self):
        self.counter_for_replay += 1
        step_size = 5
        self.counter_for_replay = self.counter_for_replay % step_size

    def train(self, episodes, num_epochs):
        for epoch in range(num_epochs):
            epoch_rewards = []
            rewards = []
            steps = 1000
            for episode in range(episodes):
                state = env.reset()
                ep_reward = 0
                state = np.reshape(state, [1, self.observation_space_dim])
                episode_frame_count = 0
                for step in range(steps):
                    #env.render()
                    #time.sleep(0.0003)
                    action = self.get_action(state)
                    next_state, reward, done, info = env.step(action)
                    next_state = np.reshape(next_state, [1, self.observation_space_dim])
                    ep_reward += reward
                    state = next_state
                    self.update_counter()
                    self.learn_and_update_weights_by_reply() 
                    self.training_frame_count += 1
                    episode_frame_count += 1
                    if done: 
                        break
                    if self.training_frame_count == 2000000:
                        break
                    if self.eps > self.eps_min:
                        self.eps *= self.eps_decay

                    last_rewards_mean = np.mean(self.rewards[-100:])
                    if last_rewards_mean > 200:
                        print("DQN Training Complete...")
                        break
                        
                rewards.append(ep_reward)
                print("Training Ep: {}, Ep Reward: {:.2f}, Last Reward: {:.2f}, Total Frames: {}, Ep Frames: {}".format(episode, ep_reward, reward, self.training_frame_count, episode_frame_count))
            epoch_rewards.append(rewards)
            env.close()
            
            for n, rewards in enumerate(epoch_rewards):
                x = range(len(rewards))
                cumsum = np.cumsum(rewards)
                avgs = [cumsum[ep]/(ep+1) if ep<100 else (cumsum[ep]-cumsum[ep-100])/100 for ep in x]
                plt.plot(x, avgs, label = "Training Epoch {}".format(epoch))
            plt.legend()
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.title("DQN Training Performance")
            plt.savefig("DQN_Training_Performance")
            

    def save(self, name):
        self.model.save(name)

    def test_trained_model(self, trained_model, num_epochs, num_episodes):
        for epoch in range(num_epochs):
            epoch_rewards = []
            rewards = []
            steps = 1000
            for episode in range(num_episodes):
                trained_state = env.reset()
                ep_reward = 0
                observation_space_dim = env.observation_space.shape[0]
                trained_state = np.reshape(trained_state, [1, observation_space_dim])
                for step in range(steps):
                    #env.render()
                    #time.sleep(0.0003)
                    trained_action = np.argmax(trained_model.predict(trained_state)[0])
                    next_state, reward, done, info = env.step(trained_action)
                    next_state = np.reshape(next_state, [1, observation_space_dim])
                    trained_state = next_state
                    ep_reward += reward
                    if done:
                        break
                rewards.append(ep_reward)
                print("Testing Episode: {}, Episode Reward: {:.2f}, Last Frame Reward: {:.3f}".format(episode, ep_reward, reward))
            epoch_rewards.append(rewards)
            env.close()

            for n, rewards in enumerate(epoch_rewards):
                x = range(len(rewards))
                cumsum = np.cumsum(rewards)
                avgs = [cumsum[ep]/(ep+1) if ep<100 else (cumsum[ep]-cumsum[ep-100])/100 for ep in x]
                plt.plot(x, avgs, label="Testing Curve")
            plt.legend()
            plt.title("DQN Performance after Training")
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.savefig("DQN_Performance_after_Training.png")

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    lr = 0.1
    eps = 1.0
    eps_decay = 0.995
    training_episodes = 200
    model = DQN(env, lr, eps, eps_decay)
    model.train(training_episodes, num_epochs=5)
    model.save("simpleDQN_trained_model.h5")
    trained_model = load_model("simpleDQN_trained_model.h5")
    model.test_trained_model(trained_model, num_epochs = 1, num_episodes=100)







        
