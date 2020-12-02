class RLAgent:
    def __init__(self, env, training_episodes, testing_episodes, training_frames, testing_frames):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.training_episodes = training_episodes
        self.testing_episodes = testing_episodes
        self.training_frames = training_frames
        self.testing_frames = testing_frames
        
        self.test_rewards = []
        self.test_episode_rewards = []
        self.test_average_rewards = []
        self.training_rewards = []
        self.training_episode_rewards = []
        self.training_average_rewards =[]
        
        
        
        