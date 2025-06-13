class PolicyDistribution:
    def __init__(self, args):
        self.args = args
        self.agent_output = None

    def select_action(self, agent_output, t_env, t, test_mode=False):
        self.agent_output = agent_output
        return agent_output.max(1)[1]