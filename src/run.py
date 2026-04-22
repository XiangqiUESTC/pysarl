import os
import pprint
import random
from os.path import abspath, dirname
from pathlib import Path

import numpy as np
import torch

from learners import REGISTER as learner_REGISTRY
from runners import REGISTRY as runner_REGISTRY
from utils.functions import build_run_name
from utils.functions import dict_to_namespace
from utils.logger import MyLogger


def set_random_seed(seed):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run(ex_run, config, log):
    args = dict_to_namespace(config)
    args.run_name = build_run_name(args.name, args.game_name, args.alg)
    set_random_seed(getattr(args, "seed", None))

    log.info("实验参数如下")
    experiment_params = pprint.pformat(config, indent=4, width=1)
    log.info("\n\n" + experiment_params + "\n")

    logger = MyLogger(args, log)

    if args.use_tensorboard:
        results_file = os.path.join(dirname(dirname(abspath(__file__))), args.local_results_path)
        logger.setup_tensorboard(results_file)

    training(args, logger)


def training(args, logger):
    args.render_test_env = bool(getattr(args, "evaluate", False) and getattr(args, "render", False))
    runner = runner_REGISTRY[args.runner](args, logger)
    learner = learner_REGISTRY[args.learner](args, runner, logger)

    if args.device == "cuda":
        learner.cuda()

    model_root = Path(dirname(dirname(abspath(__file__)))) / args.local_results_path / "models"
    save_root = model_root / args.run_name

    checkpoint_path = getattr(args, "checkpoint_path", "")
    if getattr(args, "evaluate", False) and not checkpoint_path:
        raise ValueError("evaluate=True 时必须提供 checkpoint_path。")

    if checkpoint_path:
        load_path = select_checkpoint_path(checkpoint_path, getattr(args, "load_step", 0))
        logger.logger.info(f"加载模型: {load_path}")
        learner.load_models(str(load_path))

        loaded_t_env = infer_t_env_from_checkpoint(load_path)
        if loaded_t_env is not None:
            runner.t_env = loaded_t_env

        if getattr(args, "evaluate", False):
            evaluate_only(args, runner, logger)
            return

    finish_train = False
    last_test_t = runner.t_env
    last_save_t = runner.t_env

    while runner.t_env <= args.t_max:
        runner.step()

        if learner.can_learn():
            finish_train = learner.learn()

        if runner.episode_done and (runner.t_env - last_test_t) / args.test_interval >= 1.0:
            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            runner.start_test_phase()
            test_returns = []

            for _ in range(n_test_runs):
                test_returns.append(runner.run(test_mode=True))

            mean_test_return = sum(test_returns) / len(test_returns)
            logger.log_stats("test_return_mean", mean_test_return, runner.t_env)
            logger.logger.info(f"Test t_env: {runner.t_env:>10} test_return_mean: {mean_test_return:.2f}")
            last_test_t = runner.t_env

        if args.save_model and args.save_model_interval != 0 and (runner.t_env - last_save_t) / args.save_model_interval >= 1.0:
            save_path = save_root / str(runner.t_env)
            save_path.mkdir(parents=True, exist_ok=True)
            learner.save_models(str(save_path))
            logger.logger.info(f"保存模型到 {save_path}")
            last_save_t = runner.t_env

        if finish_train:
            break

    logger.logger.info("训练结束")


def evaluate_only(args, runner, logger):
    n_test_runs = max(1, args.eval_nepisode // runner.batch_size)
    runner.start_test_phase()
    test_returns = []

    for _ in range(n_test_runs):
        test_returns.append(runner.run(test_mode=True))

    mean_test_return = sum(test_returns) / len(test_returns)
    logger.logger.info(f"Evaluate only test_return_mean: {mean_test_return:.2f}")


def select_checkpoint_path(checkpoint_path, load_step):
    checkpoint_root = Path(checkpoint_path)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint path {checkpoint_root} does not exist.")

    if (checkpoint_root / "agent.th").exists():
        return checkpoint_root

    checkpoint_dirs = [path for path in checkpoint_root.iterdir() if path.is_dir() and path.name.isdigit()]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found under {checkpoint_root}.")

    checkpoint_dirs.sort(key=lambda path: int(path.name))

    if load_step <= 0:
        return checkpoint_dirs[-1]

    return min(checkpoint_dirs, key=lambda path: abs(int(path.name) - load_step))


def infer_t_env_from_checkpoint(checkpoint_path):
    if checkpoint_path.name.isdigit():
        return int(checkpoint_path.name)

    return None
