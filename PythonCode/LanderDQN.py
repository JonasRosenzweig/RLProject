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
                    self.training_frame_count += 1
                    episode_frame_count += 1
                    if done: 
                        break
                    if self.training_frame_count == 2000000:
                        break
                    if self.eps > self.eps_min:
                        self.eps *= self.eps_decay
                        
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
    lr = 1
    eps = 1.0
    eps_decay = 0.995
    training_episodes = 3
    model = DQN(env, lr, eps, eps_decay)
    model.train(training_episodes, num_epochs=2)
    model.save("simpleDQN_trained_model.h5")
    trained_model = load_model("simpleDQN_trained_model.h5")
    model.test_trained_model(trained_model, num_epochs = 1, num_episodes=2)







        
