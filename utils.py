import json
import sys


def save_commandline(filename):
    with open(filename, 'w') as f:
        f.write('\n'.join(sys.argv[1:]))
    return


def args_to_json(args, filename):
    with open(filename, 'w') as f:
        json.dump(vars(args), f)
    return


def save_args(args, filename):
    with open(filename, 'w') as f:
        for k, v, in vars(args).items():
            f.write(str(k) + '\t' + str(v) + '\n')
    return


def join_space(space1, space2):
    return {**space1, **space2}


def to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    return


def from_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def display_args(args):
    for k, v in vars(args).items():
        print(f"{k}:\t\t\t\t{v}")
    return


def display_dataset(dataset):
    def count(mask):
        return sum(mask.tolist())
    data = dataset[0]
    print(f"{dataset.name}, training nodes: {count(data.train_mask)}, "
          f"validation nodes: {count(data.val_mask)}, "
          f"test nodes: {count(data.test_mask)}"
          )


def join_params(arch_params, hw_params):
    return [{**arch, **hw} for arch, hw in zip(arch_params, hw_params)]


def parse_rollout(pattern, space, layers=2):
    params = []
    for layer in range(layers):
        layer_parms = {}
        for i, (k, v) in enumerate(space.items()):
            layer_parms[k] = v[pattern[layer * len(space) + i]]
        params.append(layer_parms)
    return params


def build_gnn_graph(dataset, params):
    in_features = dataset[0].num_node_features
    num_classes = dataset.num_classes
    graph = []
    for i, param in enumerate(params):
        layer = {}
        is_last = (i == len(params) - 1)
        # arch parameters
        layer["in_features"] = in_features
        layer["aggregation"] = param["aggregation"]
        layer["attention"] = param["attention"]
        layer["activation"] = param["activation"]
        layer["concat"] = not is_last
        layer["heads"] = param.get("heads", 1)
        layer["out_features"] = param["hidden_features"] if is_last is False \
            else num_classes
        in_features = layer["out_features"] * layer["heads"]

        # HW parameters
        if "in_tile_size" not in param or "out_tile_size" not in param:
            layer["in_tile_size"] = None
            layer["out_tile_size"] = None
        else:
            layer["in_tile_size"] = min(
                param["in_tile_size"], layer["in_features"])
            layer["out_tile_size"] = min(
                param["out_tile_size"],
                param["hidden_features"] if is_last is False else num_classes
                )
        layer["aggr_tile_size"] = param.get("aggr_tile_size", None)
        layer["attn_tile_size"] = param.get("attn_tile_size", None)
        layer["act_tile_size"] = param.get("act_tile_size", None)

        # quantization prarameters
        layer["act_int_bits"] = param.get("act_int_bits", 0)
        layer["act_frac_bits"] = param.get("act_frac_bits", 0)
        layer["weight_int_bits"] = param.get("weight_int_bits",  0)
        layer["weight_frac_bits"] = param.get("weight_frac_bits", 0)

        graph.append(layer)

    return graph


if __name__ == "__main__":
    from dataset import cora as dataset
    params = [
        {
            'attention': 'gat', 'heads': 8, 'aggregation': 'add',
            'activation': 'relu', 'hidden_features': 16,
            'act_int_bits': 0, 'act_frac_bits': 2,
            'weight_int_bits': 2, 'weight_frac_bits': 6,
            'in_tile_size': 4, 'aggregation_tile_size': 4, 'attn_tile_size': 1
            },
        {
            'attention': 'constant', 'heads': 2, 'aggregation': 'mean',
            'activation': 'elu', 'hidden_features': 64,
            'act_int_bits': 2, 'act_frac_bits': 4,
            'weight_int_bits': 1, 'weight_frac_bits': 6,
            'in_tile_size': 3, 'aggregation_tile_size': 2, 'attn_tile_size': 1
            }
            ]
    gnn_graph = build_gnn_graph(dataset, params)
    for layer in gnn_graph:
        print(layer)
