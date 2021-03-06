class Agent:
    """
    A class used to build Reinforcement Learning algorithms.
    
    This class serves as a superclass to build more specific reinforcement 
    learning (RL) algorithms and provides the fundamental parameters found in
    all RL algorithms.
    
    Attributes
    ---------
    env : env
        The environment of a RL algorithm.
        
    config : config of parameters
        Wandb's config file for storing and tracking parameters.
        
    action_space : space
        The environment's action space.
        
    action_space_size : int
        The number of values in the action space.
        
    observation_space : space
        The environment's observation space.
        
    observation_space_size : int
        The number of values in the observation space.
    
    training_episode_rewards : list of ints
        List of rewards during training.
        
    training_average_rewards : list of ints
        List of average rewards during training.
        
    test_episode_rewards : list of floats
        List of rewards during testing.
        
    test_average_rewards : list of floats
        List of average rewards during testing.      
    """
    
    def __init__(self, env, config):
        
        self.env = env
        self.config = config
        
        self.action_space = env.action_space
        self.action_space_size = env.action_space.n
        
        self.observation_space = env.observation_space
        self.observation_space_size = env.observation_space.shape[0]

        self.training_episode_rewards = []
        self.training_average_rewards = []
        
        self.test_episode_rewards = []
        self.test_average_rewards = []
        
        
        