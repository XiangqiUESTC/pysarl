# sacred相关模块
from sacred import Experiment
from sacred.observers import FileStorageObserver

# 系统模块及工具模块
import sys
import os
from os.path import dirname, abspath, join
from copy import deepcopy
import yaml

# 框架其他模块
from utils.logger import get_logger
from utils.functions import recursive_dict_update
from utils.functions import get_config
from run import run


# 创建实验
ex = Experiment("pysarl")

@ex.main
def my_main(_run, _config, _log):
    """
        Description: sacred实验函数，main.py主要处理参数和配置，处理完成之后主要的运行在run.py的run函数中
        Arguments:
            _run: 当前实验的运行对象
            _config: 所有配置的字典
            _log: Sacred创建的日志对象
    """
    # 调用run.py中的run函数开始运行
    run(_run, _config, _log)


if __name__ == '__main__':
    # 日志记录设置
    logger = get_logger()
    ex.logger = logger

    # 根据main.py的路径（不管是相对还是绝对），获取代码根目录src绝对路径和项目根目录绝对路径
    abs_src_folder = abspath(dirname(__file__))
    abs_proj_folder = dirname(dirname(abspath(__file__)))

    # 读取默认配置
    with open(os.path.join(abs_src_folder, "config", "default.yaml")) as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # 拷贝一份命令行参数，避免后续调用get_config函数时修改了sys.argv
    params = deepcopy(sys.argv)
    # 读取环境配置
    env_config = get_config(params, "--env-config", join(abs_src_folder, "config/envs"))
    # 读取算法配置
    alg_config = get_config(params, "--alg-config", join(abs_src_folder, "config/algs"))

    # 用环境配置递归地更新默认配置
    config_dict = recursive_dict_update(config_dict, env_config)
    # 再用算法配置递归地更新默认配置
    config_dict = recursive_dict_update(config_dict, alg_config)

    # 把配置项加入实验配置中
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("实验结果将被保存在中results/sacred.")

    # 储存实验结果的文件夹
    results_path = os.path.join(abs_proj_folder, "results")
    file_obs_path = os.path.join(abs_proj_folder, "results", "sacred")

    ex.observers.append(FileStorageObserver.create(file_obs_path))

    # 开始实验
    ex.run_commandline(params)
