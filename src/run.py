import os
from os.path import abspath
from os.path import dirname
import pprint

from utils.logger import MyLogger
from utils.functions import dict_to_namespace

from runners import REGISTRY as runner_REGISTRY
from buffers import REGISTRY as buffer_REGISTRY
from learners import REGISTER as learner_REGISTRY


def run(ex_run, config, log):
    """
        Description: 在正式开始实验前再做一些准备，如装配tensorboard
        Arguments:
            ex_run: 
            config: 
            log: 
    """
    args = dict_to_namespace(config)

    log.info("实验参数为:")
    experiment_params = pprint.pformat(config,
                                       indent=4,
                                       width=1)
    log.info("\n\n" + experiment_params + "\n")

    logger = MyLogger(log)

    # 如果要使用tensorboard，在日志实例中装配
    if args.use_tensorboard:
        results_file = os.path.join(dirname(dirname(abspath(__file__))), args.local_results_path)
        logger.setup_tensorboard(results_file)

    # 开始训练!
    training(args, logger)


def training(args, logger):
    """
        Description: 封装强化学习所有流程的函数
        Arguments:
            args: args是一个SimpleNamespace对象，由sacred配置项转化而来，它控制整个强化学习流程的各个方面
            logger: logger是日志器对象
    """
    # 初始化runner
    runner = runner_REGISTRY[args.runner](args, logger)

    # 初始化buffer
    buffer = buffer_REGISTRY[args.buffer](args, runner.scheme)

    # 初始化learner
    learner = learner_REGISTRY[args.learner](args, runner.scheme, runner.controller,logger)

    if args.device == "cuda":
        learner.cuda()

    # 跑满t_max步为止
    while runner.t_env <= args.t_max:

        # runner控制env和agent交互，不同的runner有不同的控制粒度
        transaction = runner.run()

        # buffer收集数据
        buffer.insert(transaction)

        # 如果满足采样条件，就采样并学习
        if buffer.can_sample():
            # 学习方法
            learner.learn(buffer, runner.t_env, runner.episode)
