import logging
from torch.utils.tensorboard import SummaryWriter
import os
import datetime

class MyLogger:
    def __init__(self, args, logger):
        self.writer = None
        self.logger = logger
        self.args = args

    def setup_tensorboard(self, results_file):
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.writer = SummaryWriter(log_dir=os.path.join(results_file, 'tb_logs', f"{self.args.name}-{time_str}"))

    # 记录标量数据
    def log_scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
        else:
            assert False, 'Config use_tensorboard is set to False'

def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger
