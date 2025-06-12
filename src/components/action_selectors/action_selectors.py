from .epsilon_schedules.decline_then_flat import DeclineThenFlat
from torch.distributions import Categorical
import torch

REGISTRY = {}


class EpsilonGreedyActionSelector:
    def __init__(self, args):
        self.args = args

        self.epsilon_schedule = DeclineThenFlat(args.epsilon_start, args.epsilon_anneal_time,
                                                min_epsilon=args.epsilon_finish,
                                                decay="linear")

        self.epsilon = self.epsilon_schedule.eval(0)

    def select_action(self, agent_output, t_env, t, test_mode=False):
        # if in test,no explore
        if test_mode:
            self.epsilon = 0
        else:
            self.epsilon = self.epsilon_schedule.eval(t_env)

        # greedy action select strategy
        rand_num = torch.rand(*(agent_output.shape[0:-1]))

        explore = (rand_num < self.epsilon).long()

        action_num = agent_output.shape[-1]

        # uniform sampling
        probabilities = torch.ones(action_num)
        distribution = Categorical(probabilities)

        random_action = distribution.sample(explore.shape)
        max_q_action = agent_output.max(dim=-1)[1]

        picked_actions = random_action * explore + (1 - explore) * max_q_action
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
