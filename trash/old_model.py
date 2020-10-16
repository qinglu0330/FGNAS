import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


def quantize(x, int_bits=None, frac_bits=None, signed=True):
    if int_bits is None or frac_bits is None:
        # no quantization
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


# Implementation following original papers
class Attention(nn.Module):

    def __init__(self, attention_type, aggregation_type, in_features,
                 out_features, heads=1, concat=True):
        super().__init__()
        assert attention_type in ("constant", "gcn", "gat")
        if attention_type == "constant":
            self.op = ConstantConv(in_features, out_features)
        elif attention_type == "gcn":
            self.op = GCNConv(in_features, out_features)
        else:
            self.op = GATConv(in_features, out_features,
                              heads=heads, concat=concat)
        assert aggregation_type in ("add", "mean", "max")
        self.op.aggr = aggregation_type

        self.heads = heads
        self.concat = concat

    def forward(self, x, edge_index):
        x = self.op(x, edge_index)
        if isinstance(self.op, GATConv):
            return x
        if self.concat is True:
            return x.repeat(1, self.heads)
        else:
            return x

    def quantize(self, int_bits=None, frac_bits=None):
        for params in self.op.parameters():
            params.data.copy_(quantize(params.data, int_bits, frac_bits))


class Layer(nn.Module):

    def __init__(self, in_features, hidden_features, attention, aggr, heads,
                 activation, concat=True, out_features=None,
                 act_int_bits=None, act_frac_bits=None,
                 weight_int_bits=None, weight_frac_bits=None,  **kwargs):
        super().__init__()
        if out_features is None:
            out_features = hidden_features
        self.attention = Attention(attention, aggr, in_features, out_features,
                                   heads=heads, concat=concat)
        self.activation = Activation(activation)
        self.act_int_bits = act_int_bits
        self.act_frac_bits = act_frac_bits
        self.weight_int_bits = weight_int_bits
        self.weight_frac_bits = weight_frac_bits

    def forward(self, x, edge_index):
        # quantize input features
        x = quantize(x, self.act_int_bits, self.act_frac_bits)
        # quantize weight of the current layer
        self.attention.quantize(self.weight_int_bits, self.weight_frac_bits)
        x = self.attention(x, edge_index)
        return self.activation(x)


class Net(nn.Module):

    def __init__(self, input_features, num_classes, params=[]):
        super().__init__()
        self.layers = nn.ModuleList()
        in_features = input_features
        for i, param in enumerate(params):
            is_last = (i == len(params) - 1)
            self.layers.append(
                Layer(in_features, concat=not is_last,
                      out_features=num_classes if is_last is True else None,
                      **param)
            )
            in_features = param["hidden_features"] * param["heads"]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


if __name__ == "__main__":
    arch_parameters = [
        {"hidden_features": 16, "attention": "gcn", "heads": 1, "aggr": "add",
         "activation": "relu", "act_int_bits": 6, "act_frac_bits": 6,
         "weight_int_bits": 6, "weight_frac_bits": 6},
        {"hidden_features": "*", "attention": "gcn", "heads": 1,
         "aggr": "mean", "activation": "none", "act_int_bits": 6,
         "act_frac_bits": 6, "weight_int_bits": 6, "weight_frac_bits": 6},
    ]
    from dataset import cora
    from fitter import Fitter

    data = cora[0]
    model = Net(
        cora.num_node_features, cora.num_classes, arch_parameters)
    print(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    fitter = Fitter(model, data, optimizer)
    fitter.run()
    print(model)
