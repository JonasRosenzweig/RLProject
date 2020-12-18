from Run import Run, Train


class HPSearch(Run):
    def __init__(self, Agent, run_config):
        Run.__init__(self, Agent, run_config)
        def gammaExperiment(Run):
            print("Running Experiment for Gamma")
            for episode in range(run_config['experiment_episodes']):
                
                for gamma in run_config['gammas']:
                    print("Training for Gamma: {}".format(gamma))
                    Train(HPSearch.Agent, HPSearch.run_config)
                    HPSearch.gamma_rewards.append(HPSearch.Agent.training_episode_rewards)
                    
    def learningRateExperiment():
            print("Running Experiment for Learning Rate")
            for episode in range(HPSearch.run_config['experiment_episodes']):
                
                for learning_rate in HPSearch.run_config['learning_rates']:
                    print("Training for Learning Rate: {}".format(learning_rate))
                    Train(HPSearch.Agent, HPSearch.run_config)
                    HPSearch.learning_rate_rewards.append(HPSearch.Agent.training_episode_rewards)
                    
    def epsilonDecayExperiment():
            print("Running Experiment for Epsilon Decay")
            for episode in range(HPSearch.run_config['experiment_episodes']):
                
                for epsilon_decay in HPSearch.run_config['epsilon_decays']:
                    print("Training for Learning Rate: {}".format(epsilon_decay))
                    Train(HPSearch.Agent, HPSearch.run_config)
                    HPSearch.epsilon_decay_rewards.append(HPSearch.Agent.training_episode_rewards)
    
                
                