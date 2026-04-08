import os
import pprint
from os.path import abspath
from os.path import dirname

from learners import REGISTER as learner_REGISTRY
from runners import REGISTRY as runner_REGISTRY
from utils.functions import dict_to_namespace
from utils.logger import MyLogger


def run(ex_run, config, log):
    """
    在正式开始实验前准备日志和配置，然后进入训练流程。
    """
    args = dict_to_namespace(config)

    log.info("实验参数如下")
    experiment_params = pprint.pformat(config, indent=4, width=1)
    log.info("\n\n" + experiment_params + "\n")

    logger = MyLogger(args, log)

    if args.use_tensorboard:
        results_file = os.path.join(dirname(dirname(abspath(__file__))), args.local_results_path)
        logger.setup_tensorboard(results_file)

    training(args, logger)


def training(args, logger):
    """
    封装强化学习训练主循环。
    """
    # 初始化runner,runner会初始化env和agent还有buffer
    runner = runner_REGISTRY[args.runner](args, logger)

    # 初始化learner
    learner = learner_REGISTRY[args.learner](args, runner, logger)

    if args.device == "cuda":
        learner.cuda()

    # 初始化其他变量，如训练结束标志，测试计数
    finish_train = False
    last_test_t = 0

    # 跑满t_max步为止
    while runner.t_env <= args.t_max:

        # runner控制env和agent交互，不同的runner有不同的控制粒度
        runner.step()

        if learner.can_learn():
            finish_train = learner.learn()

        # 进行测试
        if runner.episode_done and (runner.t_env -  last_test_t) / args.test_interval >=1.0:
            # 计算测试次数
            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            # 开始测试
            for _ in range(n_test_runs):
                runner.run(test_mode=True)
            last_test_t = runner.t_env

        # 进行模型的保存
        if args.save_model_interval!= 0 and (runner.t_env - last_test_t) / args.save_model_interval >= 1.0:
            pass

        # 进行训练数据和测试数据的打印
        if (runner.t_env - last_test_t) / args.log_interval >= 1.0:
            pass

        if finish_train:
            break

    logger.logger.info("训练结束")
