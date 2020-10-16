
# import json
import torch
from dataset import Dataset
import utils
from model import GNN
from fitter import Fitter


def test_result(dataset, res_list, device, logger, layers=2, portion=1):
    test_trials = 10
    topk = 10

    runs = len(res_list)
    best_rewards = []
    runtime = []
    max_reward = -1
    for i in range(runs):
        df = res_list[i]
        data = df.head(int(df.shape[0] * portion))
        best_reward = data.reward.max()
        best_rewards.append(best_reward)
        if best_reward > max_reward:
            max_reward = best_reward
            best_run = i
        runtime.append(data.time.sum())
    logger.info("evaluate the search result...")
    logger.info(f"Run {best_run}: best reward: {best_rewards[best_run]}, "
                f"runtime: {runtime[best_run]}")
    logger.info(
        f"Avg. Best Reward is {sum(best_rewards) / len(best_rewards)}.")
    logger.info(
        f"a total of {df[df.time > 0.1].shape[0]} samples are trained, "
        f"and their averaged runtime is {df[df.time > 0.1].time.mean()}")

    # test the best samples
    df = res_list[best_run]
    data = df.head(int(df.shape[0] * portion))
    # best_sample = data.reward.idxmax()
    best_samples = data.nlargest(topk, "reward")
    best_samples = [index for index, row in best_samples.iterrows()]

    def get_params(row, data):
        res = []
        for i in range(layers):
            # layer = json.loads(data.loc[row][f"Layer {i}"].replace("'", '"'))
            layer = data.loc[row][f"Layer {i}"]
            res.append(layer)
        return res

    # data.loc[best_samples[0]]
    test_accs = []
    for sample in best_samples:
        params = get_params(sample, data)
        logger.info(f"for reward: {data.loc[sample].reward}...")
        best_acc = 0
        for i in range(test_trials):
            test_acc = evaluate(params, dataset, device, val_test='val')
            best_acc = max(best_acc, test_acc)
        logger.info(
            f"after {test_trials} trials, the best test accuracy is {best_acc}"
            )
        test_accs.append(best_acc)
    logger.info(f"after examing the top {topk} samples, "
                f"the best accuracy is {max(test_accs)}, and "
                f"the averaged accuracy is {sum(test_accs) / topk}")
    return


def evaluate(params, dataset, device='cuda:0', val_test='test'):
    data = Dataset(dataset)
    gnn_graph = utils.build_gnn_graph(data, params)
    model = GNN(gnn_graph).to(device)
    # logger.info(dataset)
    setting = utils.from_json("json/setting.json")[dataset]
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=setting["learning_rate"],
            weight_decay=setting["weight_decay"])
    fitter = Fitter(model, data[0].to(device), optimizer)
    history = fitter.run(val_test=val_test, verbose=False)
    return max(history.val.acc)
