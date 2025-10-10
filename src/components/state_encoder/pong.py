import logging

from torch import nn


class Pong(nn.Module):
    def __init__(self, args, scheme):
        super(Pong, self).__init__()
        self.args = args
        self.scheme = scheme

        # 去掉以下代码
        self.logger = logger = logging.getLogger("my_main")
        logger.info(scheme.continuous_state_shape)

    def forward(self, states):
        self.float()
        return states