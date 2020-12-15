from Run import Run, Train



class gammaExperiment(Run):
    def __init__(self, Agent, run_config):
        print("Running Experiment for Gamma")
        Run.__init__(self, Agent, run_config)
        for episode in range(run_config['experiment_episodes']):
            
            for gamma in run_config['gammas']:
                print("Training for Gamma: {}".format(gamma))
                Train(Agent, run_config)
                self.gamma_rewards.append(Agent.training_episode_rewards)
                
class learningRateExperiment(Run):
    def __init__(self, Agent, run_config):
        print("Running Experiment for Learning Rate")
        Run.__init__(self, Agent, run_config)
        for episode in range(run_config['experiment_episodes']):
            
            for learning_rate in run_config['learning_rates']:
                print("Training for Learning Rate: {}".format(learning_rate))
                Train(Agent, run_config)
                self.learning_rate_rewards.append(Agent.training_episode_rewards)
                
class epsilonDecayExperiment(Run):
    def __init__(self, Agent, run_config):
        print("Running Experiment for Epsilon Decay")
        Run.__init__(self, Agent, run_config)
        for episode in range(run_config['experiment_episodes']):
            
            for epsilon_decay in run_config['epsilon_decays']:
                print("Training for Learning Rate: {}".format(epsilon_decay))
                Train(Agent, run_config)
                self.epsilon_decay_rewards.append(Agent.training_episode_rewards)
    
                
                