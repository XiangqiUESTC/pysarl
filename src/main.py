"""
pysarl 程序入口。
"""
import logging
from cgitb import handler

from utils.logger import get_logger

# sacred 相关模块。
from sacred import Experiment
from sacred.observers import FileStorageObserver

# 系统模块和工具模块。
import sys
import os
from os.path import dirname, abspath, join
from copy import deepcopy
import yaml

# 框架内部模块。
from utils.functions import recursive_dict_update
from utils.functions import get_config
from run import run


# 创建实验。
ex = Experiment("pysarl")


@ex.main
def my_main(_run, _config, _log):
    """
    sacred 的主实验函数，负责整理配置并启动运行流程。

    参数：
        _run: 当前实验运行对象
        _config: 全部配置字典
        _log: sacred 创建的日志对象
    """
    # 配置当前函数使用的日志输出格式。
    ch = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    _log.addHandler(ch)
    _log.propagate = False
    # 调用 run.py 中的运行入口。
    run(_run, _config, _log)


if __name__ == '__main__':
    logger = get_logger()
    ex.logger = logger
    # 获取源码目录和项目根目录的绝对路径。
    abs_src_folder = abspath(dirname(__file__))
    abs_proj_folder = dirname(dirname(abspath(__file__)))

    # 读取默认配置。
    with open(os.path.join(abs_src_folder, "config", "default.yaml")) as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # 复制一份命令行参数，避免后面修改到 sys.argv。
    params = deepcopy(sys.argv)
    # 读取环境配置。
    env_config = get_config(params, "--env-config", join(abs_src_folder, "config/envs"))
    # 读取算法配置。
    alg_config = get_config(params, "--alg-config", join(abs_src_folder, "config/algs"))

    # 预先确定游戏名称。
    game = env_config["env_args"]["game_name"]
    # 检查命令行里是否覆盖了游戏名。
    for param in params:
        splits = param.split("=")
        if splits[0] == "env_args.game_name":
            game = splits[1]
    # 读取游戏配置。
    try:
        game_config = yaml.safe_load(open(f"{abs_src_folder}/config/games/{game}.yaml"))
    except yaml.YAMLError as exc:
        assert False, f"Reading {abs_src_folder}/config/{game}.yaml error {exc}"

    # 依次把环境、游戏和算法配置合并进默认配置。
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, game_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # 把合并后的配置注册到实验对象。
    ex.add_config(config_dict)

    # 默认把 sacred 结果写入磁盘。
    logger.info("瀹為獙缁撴灉灏嗚淇濆瓨鍦ㄤ腑results/sacred.")

    # 创建实验结果目录。
    results_path = os.path.join(abs_proj_folder, "results")
    file_obs_path = os.path.join(abs_proj_folder, "results", "sacred")

    ex.observers.append(FileStorageObserver.create(file_obs_path))

    # 启动实验。
    ex.run_commandline(params)
