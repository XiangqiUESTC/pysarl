"""
    一些工具函数
"""
from collections.abc import Mapping
from copy import deepcopy
from os.path import join
from types import SimpleNamespace as sn

import yaml


def recursive_dict_update(dst_dict, src_dist):
    """
        Description: 用一个字典递归地更新另外一个字典
        Arguments:
            dst_dict: 需要更新的字典
            src_dist: 提供更新内容的字典
    """
    if src_dist is not None:
        for key, value in src_dist.items():
            if isinstance(value, Mapping):
                dst_dict[key] = recursive_dict_update(dst_dict.get(key, {}), value)
            else:
                dst_dict[key] = value
    return dst_dict

def get_config(params, arg_name, subfolder):
    """
        Description: 在命令行参数中寻找args_name=xxx的项，然后去subfolder文件夹下读取对应的配置subfolder/xxx.yaml
        Arguments:
            params: 命令行参数
            arg_name: 提供更新内容的字典
            subfolder: 子配置文件夹
    """
    config_name = None
    for i, value in enumerate(params):
        if value.split("=")[0] == arg_name:
            config_name = value.split("=")[1]
            del params[i]
            break

    if config_name is not None:
        with open(join(subfolder, f"{config_name}.yaml"), "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as _exc:
                assert False, "{}.yaml error: {}".format(config_name, _exc)
        return config_dict
    else:
        return {}

def dict_to_namespace(dictionary):
    """
        Description: 递归地把一个字典的键属性转化为一个对象的实例属性
        Arguments:
            dictionary: 一个嵌套的字典
    """
    if isinstance(dictionary, dict):
        # 创建一个新的字典来存储转换后的结果
        new_dict = {}
        for key, value in dictionary.items():
            # 递归处理嵌套的字典或列表
            new_dict[key] = dict_to_namespace(value)
        # 将新字典转换为 SimpleNamespace
        return sn(**new_dict)
    elif isinstance(dictionary, list):
        # 递归处理列表中的每个元素
        return [dict_to_namespace(item) for item in dictionary]
    else:
        # 如果不是字典或列表，直接返回值
        return dictionary

def config_copy(config):
    """
        Description: 复制一个config，config里面可能是字典列表的多重嵌套
        Arguments:
            config: 一个配置项
    """
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)