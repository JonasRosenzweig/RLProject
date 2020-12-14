class RLAgent:
    """
    A class used to build RL algorithms
    
    This class serves as a superclass to build more specific reinforcement 
    learning (RL) algorithms and provides the fundamental parameters found in
    all RL algorithms.
    
    Attributes
    ---------
    env : env
        The environment of a RL algorithm
        
    action_space : space
        The environment's action space
        
    observation_space : space
        The environment's observation space
        
    training_episodes : int
        Maximum number of training episodes
        
    testing_episodes : int
        Maximum number of testing episodes
        
    frames : int
    
    test_episode_rewards : list of ints
    
    test_average_rewards : list of ints
    
    training_episode_rewards : list of ints
    
    training_average_rewards : list of ints
    """
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
        
        
        
        