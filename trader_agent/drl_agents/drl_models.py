from stable_baselines3 import A2C, PPO, DQN


class A2Cagent(A2C):
    def __init__(self, df, env, policy, agt_params, verbose=0):
        self.verbose = verbose
        self.df = df
        self.policy = policy
        self.agt_params = agt_params
        self.agt_smparams = {k: v for k, v in self.agt_params.items() if k != 'gamma'}
        self.env = env
        self.gamma = self.agt_params['gamma']
        super(A2Cagent, self).__init__(policy = self.policy, env = self.env, verbose = self.verbose,  **self.agt_smparams)



class PPOagent(PPO):

    def __init__(self, df, env, policy, agt_params, verbose=0):
        self.verbose = verbose
        self.df = df
        self.policy = policy
        self.agt_params = agt_params
        self.agt_smparams = {k: v for k, v in self.agt_params.items() if k != 'gamma'}
        self.env = env
        self.gamma = self.agt_params['gamma']
        super(PPOagent, self).__init__(policy = self.policy, env = self.env, verbose = self.verbose,  **self.agt_smparams)

class DQNagent(DQN):

    def __init__(self, df, env, policy, agt_params, verbose=0):
        self.verbose = verbose
        self.df = df
        self.policy = policy
        self.agt_params = agt_params
        self.agt_smparams = {k: v for k, v in self.agt_params.items() if k != 'gamma'}
        self.env = env
        self.gamma = self.agt_params['gamma']

        super(DQNagent, self).__init__(policy = self.policy, env = self.env, verbose = self.verbose,  **self.agt_smparams)
