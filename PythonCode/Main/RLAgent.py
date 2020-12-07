class RLAgent:
    def __init__(self, env, training_episodes, testing_episodes, frames):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.training_episodes = training_episodes
        self.testing_episodes = testing_episodes
        self.frames = frames
        
        self.test_episode_rewards = []
        self.test_average_rewards = []

        self.training_episode_rewards = []
        self.training_average_rewards = []
        
        
        
        