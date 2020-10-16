import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import config


def quantize(x, int_bits=None, frac_bits=None, signed=True):
    # precision = 1 / 2 ** 16
    # bound = 2 ** 16
    # x = torch.round(x / precision) * precision
    # return torch.clamp(x, -bound, bound - precision)
    # return x
    if int_bits is None or frac_bits is None:
        # no quantization
        return x
    if int_bits == 0 and frac_bits == 0:
        return x
    precision = 1 / 2 ** frac_bits
    x = torch.round(x / precision) * precision
    if signed is True:
        bound = 2 ** (int_bits - 1)
        return torch.clamp(x, -bound, bound-precision)
    else:
        bound = 2 ** int_bits
        return torch.clamp(x, 0, bound-precision)


def Activation(activation_type):
    assert activation_type in ('relu', 'tanh', 'sigmoid', 'elu', 'none')
    if activation_type == "relu":
        return nn.ReLU(inplace=True)
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "sigmoid":
        return nn.Sigmoid()
    elif activation_type == "elu":
        return nn.ELU(inplace=True)
    else:
        return nn.Identity()


class ConstantConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.linear(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


# Implementation following GrahpNas and AutoGNN
class NASOP(nn.Module):

    def __init__(self, in_features, out_features, aggregation, attention,
                 **kwargs):
        super().__init__()
        assert attention in ("constant", "gcn", "gat")
        if attention == "constant":
            self.op = ConstantConv(in_features, out_features)
        elif attention == "gcn":
            self.op = GCNConv(in_features, out_features)
        else:
            self.op = GATConv(
                in_features, out_features,
                dropout=config.DROPOUT
                )
        assert aggregation in ("add", "mean", "max")
        self.op.aggr = aggregation

    def forward(self, x, edge_index):
        if config.DROPOUT > 0:
            x = F.dropout(x, p=config.DROPOUT)
        x = self.op(x, edge_index)
        return x

    def quantize(self, int_bits, frac_bits):
        for p in self.op.parameters():
            p.data.copy_(
                quantize(p.data, int_bits, frac_bits)
                )


class NASLayer(nn.Module):

    def __init__(self, in_features, out_features, aggregation, attention,
                 activation, heads=1, concat=True,
                 act_int_bits=None, act_frac_bits=None,
                 weight_int_bits=None, weight_frac_bits=None,
                 **kwargs):
        super().__init__()
        self.concat = concat
        self.ops = nn.ModuleList()
        for i in range(heads):
            self.ops.append(
                NASOP(in_features, out_features, aggregation, attention)
                    )
        self.activation = Activation(activation)
        self.act_int_bits = act_int_bits
        self.act_frac_bits = act_frac_bits
        self.weight_int_bits = weight_int_bits
        self.weight_frac_bits = weight_frac_bits

    def forward(self, x, edge_index):
        # quantize input
        if self.training is False:
            x = quantize(x, self.act_int_bits, self.act_frac_bits)
        out = []
        for op in self.ops:
            # quantize weight
            if self.training is False:
                op.quantize(self.weight_int_bits, self.weight_frac_bits)
            out.append(op(x, edge_index))
        if self.concat is True:
            x = torch.cat(out, dim=1)
        else:
            x = sum(out) / len(out)
        return self.activation(x)

    # def __repr__(self):
    #     msg = (f"NASLayer({self.act_int_bits}, {self.act_frac_bits}, "
    #            f"{self.weight_int_bits}, {self.weight_frac_bits})")
    #     return msg


class GNN(nn.Module):

    def __init__(self, gnn_graph):
        super().__init__()
        self.features = nn.ModuleList()
        for layer in gnn_graph:
            self.features.append(
                NASLayer(**layer)
            )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.features:
            x = layer(x, edge_index)
        return x


if __name__ == "__main__":
    from dataset import cora
    from fitter import Fitter
    from utils import build_gnn_graph

    data = cora[0]
    params = [
        {
            'attention': 'gcn', 'heads': 1, 'aggregation': 'add',
            'activation': 'relu', 'hidden_features': 16,
            'act_int_bits': 1, 'act_frac_bits': 6,
            'weight_int_bits': 2, 'weight_frac_bits': 6,
            'tile_size': 4, 'aggregation_tile_size': 4, 'attn_tile_size': 1
            },
        {
            'attention': 'gcn', 'heads': 1, 'aggregation': 'add',
            'activation': 'none', 'hidden_features': 64,
            'act_int_bits': 2, 'act_frac_bits': 6,
            'weight_int_bits': 2, 'weight_frac_bits': 6,
            'tile_size': 3, 'aggregation_tile_size': 2, 'attn_tile_size': 1
            }
            ]
    gnn_graph = build_gnn_graph(cora, params)

    model = GNN(gnn_graph)
    print(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    fitter = Fitter(model, data, optimizer)
    fitter.run()
    print(model)
