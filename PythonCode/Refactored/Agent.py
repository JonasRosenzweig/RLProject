import abc
class Agent(abc.ABC):
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
    def __init__(self, env):
        self.env = env
        # weird KeyError from wandb, can't fix (try uncommenting and see)
        # self.actionSpace = env.action_space
        # self.observationSpace = env.observation_space
        
        
        
        self.testEpisodeRewards = []
        self.testAverageRewards = []

        self.trainingEpisodeRewards = []
        self.trainingAverageRewards = []
        
    
    
    
        
        
        
        