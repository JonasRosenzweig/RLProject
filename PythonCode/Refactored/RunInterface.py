import abc
import wandb
import time
import numpy as np
from Agent import Agent

class RunInterface(Agent, abc.ABC):
    def trainDQN(self, Agent):
        raise NotImplementedError
    def testDQN(self, Agent):
        raise NotImplementedError
    

class trainDQN(RunInterface):
    def __init__(self, Agent):
        def trainDQN(self, Agent): 
            print("started training")
            startTime = time.time()
            for episode in range(Agent.config.trainingEpisodes):
                frames = Agent.config.frames
                state = Agent.env.reset()
                episodeReward = 0
                frameCount = 0
                print("started frames")
                for frame in range(frames):
                    action = Agent.get_action(state)
                    next_state, reward, done, info = Agent.env.step(action)
                    next_state = np.reshape(next_state, [1, Agent.obsDim])
                    episodeReward += reward
                    Agent.trainingFrameCount += 1
                    frameCount += 1
                    Agent.addToMem(state, action, reward, next_state, done)
                    state = next_state
                    Agent.updateCounter()
                    Agent.learnFromMem()
                    if done:
                        break
                if Agent.epsilon > Agent.config.epsilonMin:
                    Agent.epsilon *= Agent.config.epsilonDecay
                averageReward = np.mean(Agent.trainingEpisodeRewards[-100:])
                if averageReward >= 200:
                    break
                if episode > 100 and averageReward < -200:
                    break
                trainTimeMinutes = (time.time() - startTime) / 60
                if trainTimeMinutes >= 30:
                    break
                Agent.trainingEpisodeRewards.append(episodeReward)
                Agent.trainingAverageRewards.append(averageReward)
                wandb.log({'average reward': averageReward, 'last reward': reward, 'epsilon': Agent.epsilon, 'episode': episode }, step=episode)
                print("""Episode: {}\t\t\t|| Episode Reward: {:.2f}
    Last Frame Reward: {:.2f}\t|| Average Reward: {:.2f}\t|| Epsilon: {:.2f}
    Frames this episode: {}\t\t|| Total Frames trained: {}\n"""
                    .format(episode, episodeReward, reward, averageReward, Agent.epsilon, frameCount, Agent.trainingFrameCount))
            Agent.env.close()
                
                
                
                
                
            
                             
