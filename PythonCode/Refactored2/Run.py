import numpy as np
import time
import wandb


class Run:
    def __init__(self, Agent, run_config):
        # config: episodes, steps, render, early_stop, episode_time_limit
        self.run_config = run_config
        self.Agent = Agent
       
        self.test_episode_rewards = []
        self.test_average_rewards = []

        self.training_episode_rewards = []
        self.training_average_rewards = []
        
        self.gamma_rewards = []
        self.learning_rate_rewards = []
        self.epsilon_decay_rewards = []
        
        self.run_frame_count = 0

    def train(self, goal, min_reward):
        print("Training {}".format(self.Agent.name))
        start_time = time.time()
        self.goal = goal
        self.min_reward = min_reward
        
        for episode in range(self.run_config['training_episodes']):
            
            episode_reward = 0
            episode_frame_count = 0
            state = self.Agent.env.reset()
            if self.Agent.config.name == "QAgent":
                self.state = self.Agent.discretize(state)
            elif self.Agent.config.name == "DQAgent":
                state = np.reshape(state, [1, self.Agent.observation_space_size])
                
            for step in range(self.run_config['steps']):
                action = self.Agent.act(state)
                next_state, reward, done, info = self.Agent.env.step(action)
                
                if self.Agent.config.name == "QAgent":
                    next_state = self.Agent.discretize(state)
                    self.Agent.updateQ(state, action, reward, next_state, done)
                    state = next_state
                
                if self.Agent.config.name == "DQAgent":
                    next_state = np.reshape(next_state, [1, self.Agent.observation_space_size])
                    self.Agent.addToMemory(state, action, reward, next_state, done)
                    state = next_state
                    self.Agent.updateReplayCount()
                    self.Agent.learnFromMemory()
                
                episode_reward += reward
                episode_frame_count += 1
                self.run_frame_count += 1
                
                if self.run_config['render'] == True:
                    self.Agent.env.render()
                
                if done:
                    break
                
            if self.Agent.config.epsilon > self.Agent.config.epsilon_min:
                self.Agent.config.epsilon *= self.Agent.config.epsilon_decay
            
            average_reward = np.mean(self.training_episode_rewards[-100:])
            train_time_minutes = (time.time() - start_time)/60
            
            if average_reward > self.run_config['goal']:
                break
            if average_reward < self.run_config['min_reward'] and episode > 100 and self.run_config['early_stop']:
                break
            if train_time_minutes > self.run_config['episode_time_limit'] and self.run_config['early_stop']:
                break
            self.training_episode_rewards.append(episode_reward)
            self.training_average_rewards.append(average_reward)
            wandb.log({'average reward': average_reward, 'last reward': reward, 'epsilon': self.Agent.config.epsilon, 'episode': episode }, step=episode)
            print("""Episode: {}\t\t\t|| Episode Reward: {:.2f}
Last Frame Reward: {:.2f}\t|| Average Reward: {:.2f}\t|| Epsilon: {:.2f}
Frames this episode: {}\t\t|| Total Frames trained: {}\n"""
                .format(episode, episode_reward, reward, average_reward, self.Agent.config.epsilon, episode_frame_count, self.run_frame_count))
        self.Agent.env.close()
        
    def test(self, trained_model):
        print("Testing {}".format(self.Agent.name))
        self.trained_model = trained_model
        
        for episode in range(self.run_config.episodes):
            episode_reward = 0
            state = self.Agent.env.reset()
            if self.Agent.config.name == "QAgent":
                pass
            if self.Agent.config.name == "DQAgent":
                state = np.reshape(state, [1, self.Agent.observation_space_size])
            
            for step in range(self.run_config.steps):
                if self.Agent.config.name == "QAgent":
                    pass
                if self.Agent.config.name == "DQAgent":
                    action = np.argmax(trained_model.predict(state)[0])
                    next_state, reward, done, info = self.Agent.env.step(action)
                    next_state = np.reshape(state, [1, self.Agent.observation_space_size])
                    state = next_state
                episode_reward += reward
                
                if done: 
                    break
            
            average_reward_trained = np.mean(self.trained_rewards[-100:])
            self.test_episode_rewards.append(episode_reward)
            self.test_average_rewards.append(average_reward_trained)
            print("""Episode: {}\t\t\t|| Episode Reward: {:.2f}\
Last Frame Reward: {:.2f}\t|| Average Reward: {:.2f}"""
              .format(episode, episode_reward, reward, average_reward_trained))
        self.Agent.env.close()
        
class Train(Run):
    def __init__(self, Agent, run_config, goal, min_reward):
        print("Training {}".format(Agent.name))
        Run.__init__(self, Agent, run_config)
        start_time = time.time()
        self.goal = goal
        self.min_reward = min_reward
        
        for episode in range(run_config['training_episodes']):
            
            episode_reward = 0
            episode_frame_count = 0
            state = Agent.env.reset()
            if Agent.config.name == "QAgent":
                state = Agent.discretize(state)
            elif Agent.config.name == "DQAgent":
                state = np.reshape(state, [1, Agent.observation_space_size])
                
            for step in range(run_config['steps']):
                action = Agent.act(state)
                next_state, reward, done, info = Agent.env.step(action)
                
                if Agent.config.name == "QAgent":
                    next_state = Agent.discretize(state)
                    Agent.updateQ(state, action, reward, next_state, done)
                    state = next_state
                
                if Agent.config.name == "DQAgent":
                    next_state = np.reshape(next_state, [1, Agent.observation_space_size])
                    Agent.addToMemory(state, action, reward, next_state, done)
                    state = next_state
                    Agent.updateReplayCount()
                    Agent.learnFromMemory()
                
                episode_reward += reward
                episode_frame_count += 1
                self.run_frame_count += 1
                
                if run_config['render'] == True:
                    Agent.env.render()
                
                if done:
                    break
                
            if Agent.config.epsilon > Agent.config.epsilon_min:
                Agent.config.epsilon *= Agent.config.epsilon_decay
            
            average_reward = np.mean(self.training_episode_rewards[-100:])
            train_time_minutes = (time.time() - start_time)/60
            
            if average_reward > run_config['goal']:
                break
            if average_reward < run_config['min_reward'] and episode > 100 and run_config['early_stop']:
                break
            if train_time_minutes > run_config['episode_time_limit'] and run_config['early_stop']:
                break
            self.training_episode_rewards.append(episode_reward)
            self.training_average_rewards.append(average_reward)
            wandb.log({'average reward': average_reward, 'last reward': reward, 'epsilon': Agent.config.epsilon, 'episode': episode }, step=episode)
            print("""Episode: {}\t\t\t|| Episode Reward: {:.2f}
Last Frame Reward: {:.2f}\t|| Average Reward: {:.2f}\t|| Epsilon: {:.2f}
Frames this episode: {}\t\t|| Total Frames trained: {}\n"""
                .format(episode, episode_reward, reward, average_reward, Agent.config.epsilon, episode_frame_count, self.run_frame_count))
        Agent.env.close()
        
class Test(Train):
    def __init__(self, Agent, run_config, trained_model):
        print("Testing {}".format(Agent.name))
        Run.__init__(self, Agent, run_config)
        self.trained_model = trained_model
        
        for episode in range(run_config.episodes):
            episode_reward = 0
            state = Agent.env.reset()
            if Agent.config.name == "QAgent":
                pass
            if Agent.config.name == "DQAgent":
                state = np.reshape(state, [1, Agent.observation_space_size])
            
            for step in range(run_config.steps):
                if Agent.config.name == "QAgent":
                    pass
                if Agent.config.name == "DQAgent":
                    action = np.argmax(trained_model.predict(state)[0])
                    next_state, reward, done, info = Agent.env.step(action)
                    next_state = np.reshape(state, [1, Agent.observation_space_size])
                    state = next_state
                episode_reward += reward
                
                if done: 
                    break
            
            average_reward_trained = np.mean(self.trained_rewards[-100:])
            self.test_episode_rewards.append(episode_reward)
            self.test_average_rewards.append(average_reward_trained)
            print("""Episode: {}\t\t\t|| Episode Reward: {:.2f}\
Last Frame Reward: {:.2f}\t|| Average Reward: {:.2f}"""
              .format(episode, episode_reward, reward, average_reward_trained))
        Agent.env.close()
        

                
                
            
            
        
    
                
                    
                
            
           
        
        
 
    
    
    
                
                    
                    
                    
                    
                    
                    
                        
                    
                    
                
                
                
                
            
            
        