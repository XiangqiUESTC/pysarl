# sacred相关模块
from sacred import Experiment
from sacred.observers import FileStorageObserver

# 系统模块及工具模块
import sys
import os
from os.path import dirname, abspath
from copy import deepcopy
import yaml
from collections.abc import Mapping

# 框架其他模块
from utils.logging import get_logger
from run import run

# 创建实验
ex = Experiment("pysarl")

# 储存实验结果的文件夹
results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

# 日志记录设置
logger = get_logger()
ex.logger = logger


# 一些个工具函数

# 这个方法用于读取命令参数中含有
def _get_config(_params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(_params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del _params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                _config_dict = yaml.safe_load(f)
            except yaml.YAMLError as _exc:
                assert False, "{}.yaml error: {}".format(config_name, _exc)
        return _config_dict


# 这个方法用于递归地更新一个字典
def recursive_dict_update(d, u):
    if u is not None:
        for k, v in u.items():
            if isinstance(v, Mapping):
                d[k] = recursive_dict_update(d.get(k, {}), v)
            else:
                d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

@ex.main
def my_main(_run, _config, _log):
    # 开始运行
    run(_run, _config, _log)


if __name__ == '__main__':
    params = deepcopy(sys.argv)
    # 根据main.py的路径（不管是相对还是绝对），找到默认的配置文件
    with open(os.path.join(dirname(__file__), "config", "default.yaml")) as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "dqn.yaml error: {}".format(exc)

    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--alg-config", "algs")

    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("实验结果将被保存在中results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
