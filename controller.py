
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# configurations of the controller
INPUT_SIZE = 10
HIDDEN_SIZE = 100
NUM_LAYERS = 1
BATCH_SIZE = 5
LR = 0.01
BASELINE_DECAY = 0.9


# the policy network containing the trainable parameters
class PolicyNetwork(nn.Module):
    def __init__(self, pattern=(2, 2)):
        super(PolicyNetwork, self).__init__()
        self.pattern = pattern
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS)
        for i in range(len(pattern)):
            setattr(self, 'embedding_{}'.format(i),
                    nn.Embedding(pattern[(i-1) % len(pattern)], INPUT_SIZE)
                    )
            setattr(self, 'classifier_{}'.format(i),
                    nn.Linear(HIDDEN_SIZE, pattern[i])
                    )

    def sample(self, x, state, index=0):
        embedding = getattr(
            self, 'embedding_{}'.format(index))
        x = embedding(x)
        x, state = self.rnn(x, state)
        classifier = getattr(
            self, 'classifier_{}'.format(index))
        x = classifier(x)
        return x, state


class Controller():

    def __init__(self, pattern, device='cpu', optimizer=None, seed=0):
        random.seed(seed)
        self.device = device
        self.pattern = pattern
        self.policy = PolicyNetwork(pattern).to(device)
        self.optimizer = optimizer
        self.initial_h = torch.randn(
            NUM_LAYERS, 1, HIDDEN_SIZE, requires_grad=False
            ).to(device)
        self.initial_c = torch.randn(
            NUM_LAYERS, 1, HIDDEN_SIZE, requires_grad=False
            ).to(device)
        self.initial_input = torch.randint(
            pattern[-1], (1, 1), requires_grad=False
            ).to(device)
        self.ema = 0

    def forward(self):
        x = self.initial_input.repeat(1, BATCH_SIZE)
        state = (
            self.initial_h.repeat(1, BATCH_SIZE, 1),
            self.initial_c.repeat(1, BATCH_SIZE, 1)
            )
        logits, rollout = [], []
        for i, p in enumerate(self.pattern):
            y, state = self.policy.sample(x, state, i)
            pi = F.softmax(torch.squeeze(y, dim=0), dim=-1)
            action = torch.multinomial(pi, 1)
            x = action.unsqueeze(0).squeeze(-1).detach()
            logits.append(pi)
            rollout.append(action)
        return rollout, logits

    def sample(self):
        self.rollout, self.logits = self.forward()
        # print(self.rollout[0].size())
        return self._parse_rollout(self.rollout)

    def backward(self, rewards):
        E = torch.zeros(BATCH_SIZE, 1).to(self.device)
        for i in range(len(self.pattern)):
            logit = self.logits[i]
            prob = torch.gather(logit, -1, self.rollout[i])
            E += torch.log(prob)
        E = (E * rewards).sum()
        if getattr(self, 'optimizer', None) is None:
            E.backward()
            with torch.no_grad():
                for p in self.policy.parameters():
                    if p.grad is not None:
                        p += p.grad * LR
        else:
            loss = - E
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return (- E).item()

    def update(self, rewards):
        for i, r in enumerate(rewards):
            rewards[i] = r - self.ema
            self._update_ema(r)
        if isinstance(rewards, (list, tuple)):
            rewards = torch.tensor(rewards).unsqueeze(-1).to(self.device)
        return self.backward(rewards)

    @staticmethod
    def _parse_rollout(rollout):
        return torch.cat(rollout, dim=1).tolist()

    def _update_ema(self, reward):
        self.ema = BASELINE_DECAY * self.ema + (1-BASELINE_DECAY)*reward

    def random_sample(self):
        sample = []
        for _ in range(BATCH_SIZE):
            rollout = []
            for num_params in self.pattern:
                s = random.randint(0, num_params-1)
                rollout.append(s)
            sample.append(rollout)
        return sample


if __name__ == '__main__':
    pattern = (1, 2, 3, 4, 5, 6) * 2
    controller = Controller(pattern, seed=1)
    rollout = controller.random_sample()
    print(rollout)
