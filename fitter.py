
import torch
import torch.nn.functional as F
import collections


Stats = collections.namedtuple("Stats", ["loss", "acc"])
History = collections.namedtuple("History", ["train", "val"])


class Fitter():

    def __init__(self, model, data, optimizer, criterion=None):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.criterion = criterion

    def epoch_evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():
            loss, acc = self.forward(mask)
        return loss.item(), acc

    def epoch_train(self):
        self.optimizer.zero_grad()
        self.model.train()
        mask = self.data.train_mask
        loss, acc = self.forward(mask)
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc

    def run(self, epochs=200, start_epoch=1, val_test='val', verbose=True):
        history = History(
            Stats([], []),
            Stats([], [])
        )
        best_acc = 0
        for epoch in range(start_epoch, start_epoch+epochs):
            loss, acc = self.epoch_train()
            history.train.loss.append(loss)
            history.train.acc.append(acc)
            if verbose is True:
                print(f"Epoch: {epoch}/{start_epoch+epochs-1}\t"
                      f"Loss: {loss:.5f}, Acc: {acc:.2%}", end='')
            if val_test is not None:
                mask = self.data.val_mask if val_test == "val" else \
                    self.data.test_mask
                loss, acc = self.epoch_evaluate(mask)
                history.val.loss.append(loss)
                history.val.acc.append(acc)
                best_acc = max(best_acc, acc)
                if verbose is True:
                    print(f"\t Validation -- Loss: {loss:.5f}, "
                          f"Acc: {acc:.2%} ({best_acc:.2%})")
            else:
                print()
        return history

    def forward(self, mask):
        output = self.model(self.data)
        if self.criterion is not None:
            loss = self.criterion(output[mask], self.data.y[mask])
        else:
            loss = F.cross_entropy(output[mask], self.data.y[mask])
        correct = self.correction(output, self.data.y, mask)
        acc = correct / int(mask.sum())
        return loss, acc

    @staticmethod
    def correction(output, label, mask):
        _, pred = output.max(dim=-1)
        correct = int(pred[mask].eq(label[mask]).sum().item())
        return correct


if __name__ == "__main__":
    from torch_geometric.nn import GCNConv
    from dataset import cora

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(cora.num_node_features, 16)
            self.conv2 = GCNConv(16, cora.num_classes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    model = Net()
    data = cora[0]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    fitter = Fitter(model, data, optimizer)
    fitter.run()
