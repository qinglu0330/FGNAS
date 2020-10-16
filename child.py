import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(arch_paras):
    model = GNN(**arch_paras)
    return model


class GCNLayer(nn.Module):
    """ one layer of GCN """
    def __init__(self, in_features, out_features, nonlinear='relu',
                 dropout=None, bias=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        if bias is True:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
        if nonlinear == 'relu':
            self.nonlinear = nn.ReLU(inplace=True)
        else:
            self.nonlinear = nn.Identity()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity
        self._init_params()

    def forward(self, adj, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = adj @ x
        if self.bias is not None:
            x = x + self.bias
        x = self.nonlinear(x)
        return x

    def _init_params(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class GNN(nn.Module):
    """ GNN as node classification model """
    def __init__(self, hidden_features,  in_features, out_features, nonlinear,
                 dropout):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        for i, h in enumerate(hidden_features):
            self.layers.append(
                GCNLayer(in_features, h, nonlinear=nonlinear,
                         dropout=(0 if i == 0 else dropout))
                )
            in_features = h
        self.layers.append(
            GCNLayer(in_features, out_features, nonlinear=None,
                     dropout=dropout)
        )

    def forward(self, adj, x):
        for layer in self.layers:
            x = layer(adj, x)
        return F.log_softmax(x, dim=1)
