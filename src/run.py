import os
from os.path import abspath
from os.path import dirname
import pprint
from types import SimpleNamespace as SN

from utils.logging import myLogger

from runners import REGISTRY as runner_REGISTRY
from controllers import REGISTRY as controller_REGISTRY


def run(_run, _config, _log):
    args = dict_to_namespace(_config)

    _log.info("实验参数为:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    logger = myLogger(_log)

    # 如果要使用tensorboard，在日志实例中装配
    if args.use_tensorboard:
        results_file = os.path.join(dirname(dirname(abspath(__file__))), args.local_results_path)
        logger.setup_tensorboard(results_file)

    # 开始训练!
    training(args, logger)


def training(args, logger):
    # 初始化runner
    runner = runner_REGISTRY[args.runner](args, logger)

    # 获取环境信息
    env_info = runner.env.get_env_info()
    # 配置scheme
    scheme = {
        "state_info": {"shape": env_info['state_shape'], "dim": env_info['state_dim']},
        "action_info": {"shape": env_info['action_shape'], "dim": env_info['action_dim']},
        "max_step": runner.env.get_max_episode_steps()
    }
    controller = controller_REGISTRY[args.controller](args, scheme)

    runner.setup(scheme, controller)

    runner.run()


# 递归地把字典的键名转化为对象的属性名
def dict_to_namespace(d):
    if isinstance(d, dict):
        # 创建一个新的字典来存储转换后的结果
        new_dict = {}
        for key, value in d.items():
            # 递归处理嵌套的字典或列表
            new_dict[key] = dict_to_namespace(value)
        # 将新字典转换为 SimpleNamespace
        return SN(**new_dict)
    elif isinstance(d, list):
        # 递归处理列表中的每个元素
        return [dict_to_namespace(item) for item in d]
    else:
        # 如果不是字典或列表，直接返回值
        return d

