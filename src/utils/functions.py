"""
    一些工具函数
"""
import datetime
import re
from collections.abc import Mapping
from types import SimpleNamespace as sn

import gymnasium as gym
import numpy as np
import torch


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

def get_cli_update_value(params, key):
    for param in params:
        if param.split("=", 1)[0] == key:
            return param.split("=", 1)[1]
    return None

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


def get_space_shape_and_type(space):
    if isinstance(space, gym.spaces.Discrete):
        return (1,), torch.int64

    if isinstance(space, gym.spaces.MultiDiscrete):
        return tuple(space.shape), torch.int64

    if isinstance(space, gym.spaces.MultiBinary):
        return tuple(space.shape), torch.int64

    if isinstance(space, gym.spaces.Tuple):
        if all(isinstance(subspace, gym.spaces.Discrete) for subspace in space.spaces):
            return (len(space.spaces),), torch.int64
        raise NotImplementedError(f"Tuple space {space} is not supported.")

    if isinstance(space, gym.spaces.Box):
        if np.issubdtype(space.dtype, np.integer):
            return space.shape, torch.int64
        return space.shape, torch.float32

    raise NotImplementedError(f"Space type {type(space)} is not supported.")


def build_run_name(base_name, game_name, alg_name, timestamp=None):
    if timestamp is None:
        timestamp = datetime.datetime.now()

    short_time = timestamp.strftime("%y-%m-%d-%H-%M-%S")
    normalized_base_name = _normalize_run_name_part(base_name)
    normalized_game_name = _normalize_run_name_part(game_name)
    normalized_alg_name = _normalize_run_name_part(alg_name)
    return f"{normalized_base_name}-{normalized_game_name}-{normalized_alg_name}-{short_time}"


def _normalize_run_name_part(value):
    normalized_value = re.sub(r"\s+", "_", str(value).strip())
    normalized_value = re.sub(r'[\\/:*?"<>|]+', "-", normalized_value)
    return normalized_value or "unknown"


def move_batch_to_device(batch, device):
    moved_batch = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value

    return moved_batch
