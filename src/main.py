"""
pysarl 程序入口。
"""
import logging
import os
import sys
from copy import deepcopy
from os.path import abspath, dirname, join

import yaml
from sacred import Experiment
from sacred.observers import FileStorageObserver

from run import run
from utils.functions import get_cli_update_value
from utils.functions import recursive_dict_update
from utils.logger import get_logger


ex = Experiment("pysarl")
ex.add_config({"alg": None})


@ex.main
def my_main(_run, _config, _log, alg):
    ch = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S")
    ch.setFormatter(formatter)
    _log.addHandler(ch)
    _log.propagate = False
    run(_run, _config, _log)


if __name__ == "__main__":
    logger = get_logger()
    ex.logger = logger

    abs_src_folder = abspath(dirname(__file__))
    abs_proj_folder = dirname(dirname(abspath(__file__)))

    with open(os.path.join(abs_src_folder, "config", "default.yaml")) as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, f"default.yaml error: {exc}"

    with open(os.path.join(abs_src_folder, "config", "algs.yaml")) as f:
        try:
            algs_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, f"algs.yaml error: {exc}"

    params = deepcopy(sys.argv)
    alg_name = get_cli_update_value(params, "alg")

    if alg_name is None:
        assert False, "必须通过 with alg=xxx 指定算法名称，例如 with alg=dqn"

    alg_config = algs_dict.get(alg_name, {})


    game = get_cli_update_value(params, "game_name")
    if game is None:
        game = config_dict["game_name"]

    try:
        with open(f"{abs_src_folder}/config/games/{game}.yaml", "r") as f:
            game_config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        assert False, f"Reading {abs_src_folder}/config/{game}.yaml error {exc}"

    if game_config is None:
        game_config = {}

    alg_args = game_config.pop("alg_args", {})
    game_alg_config = alg_args.get(alg_name, {})
    game_alg_wrappers = game_alg_config.pop("wrapper", [])

    config_dict = recursive_dict_update(config_dict, game_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    config_dict = recursive_dict_update(config_dict, game_alg_config)
    config_dict["alg_env_wrappers"] = list(game_alg_wrappers)

    ex.add_config(config_dict)

    logger.info("默认把 sacred 结果保存到 results/sacred。")

    file_obs_path = os.path.join(abs_proj_folder, "results", "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
