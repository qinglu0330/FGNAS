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
from evaluate import test_result


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
    '--fpga', action='store_true',
    help="add fpga model to the search"
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
    '-e', '--episodes', type=int, default=200,
    help="max number of episodes (default: 200)."
)
parser.add_argument(
    '--gpu-eps', type=int, default=None,
    help="max number of episodes effective on gpu (default: 200)."
)
parser.add_argument(
    '--run', type=int, default=1,
    help="number of runs (default: 1)."
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
parser.add_argument(
    "--evaluate", action='store_true',
    help="evaluate the result."
)
args = parser.parse_args()


def nas(logger):
    SPACE = space.ARCH_SPACE if args.arch_json is None else {}
    if args.quantize is True:
        SPACE = utils.join_space(SPACE, space.QUANT_SPACE)
    if args.fpga is True:
        SPACE = utils.join_space(SPACE, space.HW_SPACE)
        fpga_info = utils.from_json(args.fpga_info_json)
        fpga.load_fpga_info(fpga_info)
    if len(SPACE) == 0:
        print("Space is empty!")
        exit()
    pattern = [len(params) for params in SPACE.values()] * args.layers
    device = 'cpu' if torch.cuda.is_available() is False else \
        'cuda:{}'.format(args.gpu)
    logger.info("using device {}".format(device))
    controller = Controller(pattern, seed=args.seed)
    best_reward, child_id = 0, 0
    best_sample = 0
    # reward_history = []
    dataset = Dataset(args.dataset)
    data = dataset[0].to(device)
    # utils.display_dataset(dataset)
    result = {f"Layer {i}": [] for i in range(args.layers)}
    result["reward"] = []
    result["time"] = []
    registry = {}
    gpu_calls = 0
    if args.arch_json is not None:
        arch_params = utils.from_json(args.arch_json)

    def get_accuracy(params):
        nonlocal gpu_calls
        gpu_calls += 1
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

    def fpga_analysis(params):
        gnn_graph = utils.build_gnn_graph(dataset, params)
        fpga_model = FPGAModel(data, gnn_graph)
        return fpga_model.validate(
            Resource(args.rLUT, args.rReg, args.rDSP), args.rCycle)

    def get_reward(rollout, params):
        if args.register is True:
            key = tuple(rollout)
            if key in registry:
                return registry[key]
        if args.fpga is True and fpga_analysis(params) is False:
            reward = 0
        else:
            reward = get_accuracy(params)
        return reward

    episodes = args.episodes if args.gpu_eps is None else 2 ** 32
    for eps in range(episodes):
        logger.info(f"Episode {eps+1}...")
        if args.random is True:
            rollout = controller.random_sample()
        else:
            rollout = controller.sample()
        reward_batch = []
        for r in rollout:
            start_time = time.time()
            child_id += 1
            params = parse_rollout(r, SPACE)
            if args.arch_json is not None:
                params = utils.join_params(params, arch_params)
            reward = get_reward(r, params)
            [result[f"Layer {i}"].append(params[i])
                for i in range(len(params))]
            result["reward"].append(reward)
            if reward > best_reward:
                best_reward = reward
                best_sample = child_id
            reward_batch.append(reward)
            elasped_time = time.time() - start_time
            result["time"].append(elasped_time)
            logger.info(f"Child Network: {child_id} (best: {best_sample}), "
                        # f"Rollout: {params}, "
                        f"Reward: {reward:.4f} ({best_reward:.4f}), "
                        f"Time: {elasped_time:.2f} s, "
                        f"GPU Calls: {gpu_calls}")
            registry[tuple(r)] = reward
        if args.gpu_eps is not None and gpu_calls >= args.gpu_eps:
            # print(args.gpu_eps, gpu_calls)
            break
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
    start = time.time()
    for i in range(args.run):
        logger.info("=" * 50 + f"Run {i+1}" + "=" * 50)
        res = nas(logger)
        res_list.append(res)
    runnint_time = time.time()-start
    result = pd.concat(
            res_list, axis=0,
            keys=[f"Run {i}" for i in range(len(res_list))]
            )

    if args.save is True:
        os.makedirs(work_dir, exist_ok=False)
        utils.save_args(args, os.path.join(work_dir, 'args.txt'))
        file_name = os.path.join(work_dir, "result.csv")

        result.to_csv(file_name, index=True)
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

    device = 'cpu' if torch.cuda.is_available() is False else \
        'cuda:{}'.format(args.gpu)
    if args.evaluate is True:
        test_result(
            args.dataset, res_list,
            device, logger, args.layers
            )
    logger.info(f"Total running time {runnint_time}")
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


if __name__ == "__main__":
    # print(args.gpu_eps)
    # exit()
    main()
