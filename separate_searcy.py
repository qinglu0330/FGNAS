import argparse
import matplotlib.pyplot as plt
import datetime
import os
import pandas as pd
import logging
import torch
import sys
from fitter import Fitter
from model import GNN
import space
from controller import Controller
from dataset import Dataset
import utils
import fpga
from fpga.architecture import FPGAModel
from fpga.operator import Resource
import time
import config


parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', "--dataset", type=str, default='Cora',
    help="target dataset from 'Cora' (default), 'CiteSeer', 'PubMed'."
)
parser.add_argument(
    '-l', '--layers', type=int, default=2,
    help="number of layers of the child networks (default: 1)."
)
parser.add_argument(
    '--arch-json', type=str, default=None,
    help="arch parameters to be used for quantization and hardware search"
)
parser.add_argument(
    '--quantize', action='store_true',
    help="add quantization to the search"
)
parser.add_argument(
    '--fpga-info-json', type=str, default="json/fpga/fpga_info.json",
    help="json file containing specific fpga information"
)
parser.add_argument(
    '--rLUT', type=int, default=1e5,
    help="required number of LUTs"
)
parser.add_argument(
    '--rReg', type=int, default=1e5,
    help="required number of FFs"
)
parser.add_argument(
    '--rDSP', type=int, default=1000,
    help="required number of DSPs"
)
parser.add_argument(
    '--rCycle', type=int, default=1e5,
    help="required number of cycles"
)
parser.add_argument(
    '--dropout', type=float, default=0.6,
    help="drou out rate."
)
parser.add_argument(
    '--random', action='store_true',
    help="random search."
)
parser.add_argument(
    '-ae', '--arch-episodes', type=int, default=100,
    help="max number of episodes for searching architectures (default: 200)."
)
parser.add_argument(
    '-he', '--hardware-episodes', type=int, default=100,
    help="max number of episodes for searching quantization (default: 200)."
)
parser.add_argument(
    '--run', type=int, default=5,
    help="number of runs (default: 5)."
)
parser.add_argument(
    '-g', '--gpu', type=int, default=0,
    help="the gpu to use (default: 0)."
)
parser.add_argument(
    '-s', '--seed', type=int, default=0,
    help="randomness seed (default: 0)."
)
parser.add_argument(
    '--save', action='store_true',
    help="save the result."
)
parser.add_argument(
    '--verbose', action='store_true',
    help="verbose mode, printing training progress"
)
parser.add_argument(
    '--work-dir', type=str, default="experiment/",
    help="working directory to store the result (default: \"experiment/\")."
)
parser.add_argument(
    "--register", action="store_true",
    help="resiger the pre calculated results"
)
args = parser.parse_args()


def nas(logger):
    SPACE = space.ARCH_SPACE
    if len(SPACE) == 0:
        print("Space is empty!")
        exit()
    pattern = [len(params) for params in SPACE.values()] * args.layers
    device = 'cpu' if torch.cuda.is_available() is False else \
        'cuda:{}'.format(args.gpu)
    logger.info("using device {}".format(device))
    controller = Controller(pattern)

    # reward_history = []
    dataset = Dataset(args.dataset)
    data = dataset[0].to(device)
    # utils.display_dataset(dataset)
    result = {f"Layer {i}": [] for i in range(args.layers)}
    result["reward"] = []
    result["time"] = []

    # searching architectures
    def get_accuracy(params):
        gnn_graph = utils.build_gnn_graph(dataset, params)
        model = GNN(gnn_graph).to(device)
        setting = utils.from_json("json/setting.json")[args.dataset]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=setting["learning_rate"],
            weight_decay=setting["weight_decay"])
        fitter = Fitter(model, data, optimizer)
        history = fitter.run(verbose=args.verbose)
        reward = max(history.val.acc)
        return reward

    arch_id, best_arch, best_accuracy = 0, 0, 0
    registry = {}
    for eps in range(args.arch_episodes):
        logger.info(f"Architecture Episode {eps+1}...")
        if args.random is True:
            rollout = controller.random_sample()
        else:
            rollout = controller.sample()
        reward_batch = []
        for r in rollout:
            start_time = time.time()
            arch_id += 1
            arch_params = parse_rollout(r, SPACE)
            key = tuple(r)
            reward = registry[key] if key in registry else \
                get_accuracy(arch_params)

            [result[f"Layer {i}"].append(arch_params[i])
                for i in range(len(arch_params))]
            result["reward"].append(reward)
            if reward > best_accuracy:
                best_accuracy = reward
                best_arch = arch_id
                best_arch_params = arch_params
            reward_batch.append(reward)
            elasped_time = time.time() - start_time
            result["time"].append(elasped_time)
            logger.info(f"Child Architecture: {arch_id} ",
                        f"(best: {best_arch}), "
                        f"Rollout: {arch_params}, "
                        f"Reward: {reward:.4f} ({best_accuracy:.4f}), "
                        f"Time: {elasped_time:.2f} s")
            registry[tuple(r)] = reward
        if args.random is False:
            controller.update(reward_batch)
    # searching hardware
    SPACE = space.HW_SPACE
    if args.quantize is True:
        SPACE = utils.join_space(SPACE, space.QUANT_SPACE)
    fpga_info = utils.from_json(args.fpga_info_json)
    fpga.load_fpga_info(fpga_info)
    if len(SPACE) == 0:
        print("Space is empty!")
        exit()
    pattern = [len(params) for params in SPACE.values()] * args.layers
    controller = Controller(pattern)

    def get_hw_reward(params):
        if fpga_analysis(params) is False:
            return 0
        if args.quantize is True:
            return get_accuracy(params)
        return best_accuracy

    def fpga_analysis(params):
        gnn_graph = utils.build_gnn_graph(dataset, params)
        fpga_model = FPGAModel(data, gnn_graph)
        return fpga_model.validate(
            Resource(args.rLUT, args.rReg, args.rDSP), args.rCycle)

    hw_id, best_hw, best_hw_reward = 0, 0, 0
    registry = {}
    for eps in range(args.hardware_episodes):
        logger.info(f"Hardware Episode {eps+1}...")
        if args.random is True:
            rollout = controller.random_sample()
        else:
            rollout = controller.sample()
        reward_batch = []
        for r in rollout:
            start_time = time.time()
            hw_id += 1
            hw_params = parse_rollout(r, SPACE)
            params = join_params(best_arch_params, hw_params)
            key = tuple(r)
            reward = registry[key] if key in registry else \
                get_hw_reward(params)

            [result[f"Layer {i}"].append(params[i])
                for i in range(len(params))]
            result["reward"].append(reward)
            if reward > best_hw_reward:
                best_hw_reward = reward
                best_hw = hw_id
            reward_batch.append(reward)
            elasped_time = time.time() - start_time
            result["time"].append(elasped_time)
            logger.info(f"Child Hardware: {hw_id} ",
                        f"(best: {best_hw}), "
                        f"Rollout: {hw_params}, "
                        f"Reward: {reward:.4f} ({best_hw_reward:.4f}), "
                        f"Time: {elasped_time:.2f} s")
            registry[key] = reward
        if args.random is False:
            controller.update(reward_batch)

    df = pd.DataFrame(result)
    return df


def main():
    utils.display_args(args)
    time.sleep(5)
    # setting up reproducibility with selected seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # setting up the working directory and recording args
    exp_tag = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    work_dir = os.path.join(
        args.work_dir, args.dataset, exp_tag
        )
    os.makedirs("log", exist_ok=True)
    log_path = os.path.join("log", exp_tag+".log")
    logger = get_logger(log_path)
    res_list = []
    config.DROPOUT = args.dropout
    for i in range(args.run):
        logger.info("=" * 50 + f"Run {i+1}" + "=" * 50)
        res = nas(logger)
        res_list.append(res)
    if args.save is True:
        os.makedirs(work_dir, exist_ok=False)
        utils.save_args(args, os.path.join(work_dir, 'args.txt'))
        result = pd.concat(
            res_list, axis=0,
            keys=[f"Run {i}" for i in range(len(res_list))]
            )
        file_name = os.path.join(work_dir, "result.csv")
        result.to_csv(file_name)
        logger.info(f"saving result to {file_name}")
        fig_name = os.path.join(work_dir, "progress.png")
        plt.figure()
        avg_reward = result.groupby(
            result.index.get_level_values(1))["reward"].mean()
        plt.plot(avg_reward)
        plt.title("Best reward in {} runs is {:.4f}".format(
            args.run, result["reward"].max()))
        logger.info(f"saving figure to {fig_name}")
        plt.savefig(fig_name)
        logger.info("Best reward in {} runs is {:.4f}".format(
                args.run, result["reward"].max()))
    logger.info("*" * 50 + "End" + "*" * 50)


def get_logger(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(filename, mode='w')
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S"
        )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def parse_rollout(pattern, space):
    params = []
    for layer in range(args.layers):
        layer_parms = {}
        for i, (k, v) in enumerate(space.items()):
            layer_parms[k] = v[pattern[layer * len(space) + i]]
        params.append(layer_parms)
    return params


def join_params(arch_params, hw_params):
    return [{**arch, **hw} for arch, hw in zip(arch_params, hw_params)]


if __name__ == "__main__":
    main()
