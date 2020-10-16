import matplotlib.pyplot as plt
import random
import time
import torch
import controller as ctrl


VARIANCE = 0.1


def get_target(pattern):
    target = []
    for p in pattern:
        target.append(random.randint(0, p-1))
    return target


def get_reward(rollout, target):
    error = 0
    for r, t in zip(rollout, target):
        error += abs(r != t)
    max_error = len(rollout)
    reward = 1 - error / max_error
    if VARIANCE > 0:
        reward += random.gauss(0, VARIANCE)
    return reward


def plot(reward_history):
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()


def main(pattern):
    max_epochs = 140
    controller = ctrl.Controller(pattern)
    best_reward = -100000
    start = time.time()
    target = get_target(pattern)
    reward_history = []
    for e in range(max_epochs):
        rollout = controller.sample()
        rewards = []
        for r in rollout:
            reward = get_reward(r, target)
            rewards.append(reward)
            reward_history.append(reward)
            if reward > best_reward:
                best_reward = reward
                best_rollout = r
            print("action: {}, reward: {}".format(r, reward))
        controller.update(rewards)
        print("epoch {}".format(e))
        print(f"best rollout {best_rollout}, " +
              f"best reward: {best_reward}")
    print("elasped time is {}".format(time.time()-start))
    print("target: {}".format(target))
    plot(reward_history)


if __name__ == '__main__':
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    pattern = (10,) * 20
    VARIANCE = 0.00
    ctrl.BATCH_SIZE = 5
    main(pattern)
