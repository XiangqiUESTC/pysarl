import torch
from torch.distributions import Categorical
import numpy as np

def t1():
    pass
    # q = torch.rand(5, 5, 5)
    #
    # epsilon = 1
    #
    # rand_num = torch.rand(*q.shape[0:-1], 1)
    #
    # explore = (rand_num < epsilon).long()
    #
    # action_num = q.shape[-1]
    #
    # probabilities = torch.ones(action_num)
    # distribution = Categorical(probabilities)
    #
    # random_action = distribution.sample(explore.shape)
    #
    #
    # max_q_action = q.max(dim=-1)[1].unsqueeze(-1)
    #
    #
    # # print(random_action.shape)
    # # print(explore.shape)
    # # print(max_q_action.shape)
    #
    # picked_actions = random_action * explore + (1-explore) * max_q_action
    # print(q)
    # print(picked_actions)
    # # print(picked_actions.shape)


def t2():
    a = np.array([5,6,8])
    print(np.prod(a))


def t3():
    a = torch.rand(20, 20)
    b = torch.rand(10, 12, 20, 20)
    a_dim = a.ndimension()
    c = b.flatten(start_dim=-a_dim)
    print(c.shape)


t3()
