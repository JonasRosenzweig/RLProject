class Agent:
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
    
    test_episode_rewards : list of floats
    
    test_average_rewards : list of floats
    
    training_episode_rewards : list of ints
    
    training_average_rewards : list of ints
    """
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        self.action_space = env.action_space
        self.action_space_size = env.action_space.n
        
        self.observation_space = env.observation_space
        self.observation_space_size = env.observation_space.shape[0]
        
        self.test_episode_rewards = []
        self.test_average_rewards = []

        self.training_episode_rewards = []
        self.training_average_rewards = []
        
        
        self.replay_counter = 0
        
        
        
        
        
        