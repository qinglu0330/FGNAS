import math
from .operator import Adder, Multiplier, Divider
from .operator import ReLU, Tanh, ELU, Sigmoid, Identity
from .operator import Accumulator, MaxAccumulator
from .operator import GenericMACArray, MulAdderTreeArray
from .fpga import Resource, get_fpga_capacity


class Linear():
    _type = "Linear"

    def __init__(self, in_features, out_features, in_tile_size, out_tile_size,
                 act_int_bits=0, act_frac_bits=0, weight_int_bits=0,
                 weight_frac_bits=0):
        cycle = 0
        in_tile_size = min(in_features, in_tile_size)
        if in_tile_size == in_features:
            operator = MulAdderTreeArray(
                out_tile_size, in_features,
                act_int_bits=act_int_bits,
                act_frac_bits=act_frac_bits,
                weight_int_bits=weight_int_bits,
                weight_frac_bits=weight_frac_bits
                )
        else:
            operator = GenericMACArray(
                out_tile_size, in_tile_size,
                act_int_bits=act_int_bits,
                act_frac_bits=act_frac_bits,
                weight_int_bits=weight_int_bits,
                weight_frac_bits=weight_frac_bits
                )
        self.usage = operator.usage
        cycle = (operator.cycle + 1) * math.ceil(in_features / in_tile_size)
        cycle = (cycle + 1) * math.ceil(out_features / out_tile_size)
        self.cycle = cycle
        self.out_bits = operator.out_bits


class Aggregator():

    def __init__(self, aggr_type, feature_size, tile_size):
        self.type = aggr_type
        self.usage *= tile_size
        self.cycle = (self.cycle + 1) * math.ceil(feature_size / tile_size)


class Add(Aggregator):
    _type = "Add"

    def __init__(self, data, feature_size, tile_size,
                 int_bits=0, frac_bits=0):
        accumulator = Accumulator(int_bits, frac_bits)
        usage = accumulator.usage
        cycle = accumulator.cycle
        cycle = (cycle + 1) * data.num_edges * 2 + data.num_nodes
        self.usage = usage
        self.cycle = cycle
        super().__init__(Add._type, feature_size, tile_size)
        self.out_bits = accumulator.out_bits


class Mean(Aggregator):
    _type = "Mean"

    def __init__(self, data, feature_size, tile_size, int_bits=0,
                 frac_bits=0):
        accumulator = Accumulator(int_bits, frac_bits)
        divider = Divider(*accumulator.out_bits)
        usage = accumulator.usage + divider.usage
        acc_cycle = accumulator.cycle
        div_cycle = divider.cycle
        cycle = (acc_cycle + 1) * data.num_edges * 2 + \
            (div_cycle + 1) * data.num_nodes
        self.usage = usage
        self.cycle = cycle
        super().__init__(Mean._type, feature_size, tile_size)
        self.out_bits = divider.out_bits


class Max(Aggregator):
    _type = "Max"

    def __init__(self, data, feature_size, tile_size, int_bits=0,
                 frac_bits=0):
        max_accumulator = MaxAccumulator(int_bits, frac_bits)
        usage = max_accumulator.usage
        cycle = max_accumulator.cycle
        cycle = (cycle + 1) * data.num_edges * 2 + data.num_nodes
        self.usage = usage
        self.cycle = cycle
        super().__init__(Max._type, feature_size, tile_size)
        self.out_bits = max_accumulator.out_bits


class Attention():

    def __init__(self, attn_type, usage, cycle):
        self.type = attn_type
        self.usage = usage
        self.cycle = cycle


class GCN(Attention):
    _type = "GCN"

    def __init__(self, *args, **kwargs):
        super().__init__(GCN._type, Resource(0, 0, 0), 0)


def gather(data, feature_size, tile_size, bit_width):
    divider = Divider(bit_width)
    usage = divider.usage * tile_size
    cycle = divider.cycle
    cycle = (cycle + 1) * math.ceil(feature_size / tile_size)
    cycle = (cycle + 1) * data.num_edges * 2 + 1 * data.num_nodes
    return usage, cycle


class Constant(Attention):
    _type = "Constant"

    def __init__(self, data, feature_size, attn_tile_size, int_bits=0,
                 frac_bits=0):
        divider = Divider(int_bits, frac_bits)
        usage = divider.usage * attn_tile_size
        cycle = divider.cycle
        cycle = (cycle + 1) * math.ceil(feature_size / attn_tile_size)
        cycle = (cycle + 1) * data.num_edges * 2 + 1 * data.num_nodes
        super().__init__(Constant._type, usage, cycle)
        self.out_bits = divider.out_bits


class GAT(Attention):
    _type = "GAT"

    def __init__(self, data, feature_size, attn_tile_size,
                 act_int_bits=0, act_frac_bits=0, weight_int_bits=0,
                 weight_frac_bits=0):
        linear = Linear(
            feature_size, 1, attn_tile_size, 1,
            act_int_bits=act_int_bits,
            act_frac_bits=act_frac_bits,
            weight_int_bits=weight_int_bits,
            weight_frac_bits=weight_frac_bits
            )
        usage = linear.usage * 2
        cycle = linear.cycle

        adder = Adder(*linear.out_bits)
        usage += adder.usage
        cycle += adder.cycle
        cycle = (cycle + 1) * data.num_edges * 2 + 1 * data.num_nodes
        int_bits, frac_bits = adder.out_bits

        multiplier = Multiplier(
            int_bits, frac_bits, act_int_bits, act_frac_bits)
        usage += multiplier.usage
        mul_cycle = multiplier.cycle
        mul_cycle = (mul_cycle + 1) * math.ceil(feature_size / attn_tile_size)
        mul_cycle = (mul_cycle + 1) * data.num_edges * 2 + \
            1 * data.num_nodes
        cycle += mul_cycle
        super().__init__(GAT._type, usage, cycle)
        # need truncation
        self.out_bits = (act_int_bits, act_frac_bits)


class Nonlinear():

    def __init__(self, act_type, data, feature_size, act_tile_size,
                 usage, cycle):
        self.type = act_type
        self.usage = usage
        cycle = (cycle + 1) * math.ceil(feature_size / act_tile_size)
        self.cycle = (cycle + 1) * data.num_nodes


class RelULayer(Nonlinear):
    _type = "relu_layer"

    def __init__(self, data, feature_size, act_tile_size, int_bits=0,
                 frac_bits=0):
        act = ReLU(int_bits, frac_bits)
        usage = act.usage * act_tile_size
        cycle = (act.cycle + 1) * math.ceil(feature_size / act_tile_size)
        super().__init__(RelULayer._type, data, feature_size, act_tile_size,
                         usage, cycle)
        self.out_bits = (int_bits, frac_bits)


class ELULayer(Nonlinear):
    _type = "elu_layer"

    def __init__(self, data, feature_size, act_tile_size, int_bits=0,
                 frac_bits=0):
        act = ELU(int_bits, frac_bits)
        usage = act.usage * act_tile_size
        cycle = (act.cycle + 1) * math.ceil(feature_size / act_tile_size)
        super().__init__(RelULayer._type, data, feature_size, act_tile_size,
                         usage, cycle)
        self.out_bits = (int_bits, frac_bits)


class SigmoidLayer(Nonlinear):
    _type = "sigmoid_layer"

    def __init__(self, data, feature_size, act_tile_size, int_bits=0,
                 frac_bits=0):
        act = Sigmoid(int_bits, frac_bits)
        usage = act.usage * act_tile_size
        cycle = (act.cycle + 1) * math.ceil(feature_size / act_tile_size)
        super().__init__(RelULayer._type, data, feature_size, act_tile_size,
                         usage, cycle)
        self.out_bits = (int_bits, frac_bits)


class TanhLayer(Nonlinear):
    _type = "tanh_layer"

    def __init__(self, data, feature_size, act_tile_size, int_bits=0,
                 frac_bits=0):
        act = Tanh(int_bits, frac_bits)
        usage = act.usage * act_tile_size
        cycle = (act.cycle + 1) * math.ceil(feature_size / act_tile_size)
        super().__init__(RelULayer._type, data, feature_size, act_tile_size,
                         usage, cycle)
        self.out_bits = (int_bits, frac_bits)


class Layer():

    def __init__(self, data, in_features, out_features, aggregation,
                 attention, activation, heads=1, concat=True, in_tile_size=1,
                 out_tile_size=1, aggr_tile_size=1, attn_tile_size=1,
                 act_tile_size=1, act_int_bits=0, act_frac_bits=0,
                 weight_int_bits=0, weight_frac_bits=0, **kwargs):

        usage = Resource(0, 0, 0)
        cycle = 0

        transformer = Linear(in_features, out_features, in_tile_size,
                             out_tile_size, act_int_bits, act_frac_bits,
                             weight_int_bits, weight_frac_bits)
        usage += transformer.usage
        cycle += transformer.cycle
        int_bits, frac_bits = transformer.out_bits

        assert attention in ("constant", "gcn", "gat")
        if attention == "gat":
            attentioner = GAT(
                data, out_features, attn_tile_size,
                int_bits, frac_bits, weight_int_bits, weight_frac_bits
                )
        elif attention == "gcn":
            attentioner = GCN(
                data, out_features, attn_tile_size,
                int_bits, frac_bits
                )
        else:
            attentioner = Constant(
                data, out_features, attn_tile_size,
                int_bits, frac_bits
                )
        # bit width doesn't change here
        usage += attentioner.usage
        cycle += attentioner.cycle

        assert aggregation in ("add", "mean", "max")
        if aggregation == "add":
            aggregator = Add(data, out_features, aggr_tile_size,
                             int_bits, frac_bits)
        elif aggregation == "mean":
            aggregator = Mean(data, out_features, aggr_tile_size,
                              int_bits, frac_bits)
        else:
            aggregator = Max(data, out_features, aggr_tile_size,
                             int_bits, frac_bits)
        usage += aggregator.usage
        cycle += aggregator.cycle
        int_bits, frac_bits = aggregator.out_bits

        assert activation in ('relu', 'tanh', 'sigmoid', 'elu', 'none')
        if activation == "relu":
            nonlinear = RelULayer(data, out_features, act_tile_size,
                                  int_bits=int_bits, frac_bits=frac_bits)
        elif activation == "tanh":
            nonlinear = TanhLayer(data, out_features, act_tile_size,
                                  int_bits=int_bits, frac_bits=frac_bits)
        elif activation == "sigmoid":
            nonlinear = SigmoidLayer(data, out_features, act_tile_size,
                                     int_bits=int_bits, frac_bits=frac_bits)
        elif activation == "elu":
            nonlinear = ELULayer(data, out_features, act_tile_size,
                                 int_bits=int_bits, frac_bits=frac_bits)
        else:
            nonlinear = Identity()
        usage += nonlinear.usage
        cycle += nonlinear.cycle
        int_bits, frac_bits = nonlinear.out_bits
        usage *= heads
        if concat is False and heads > 1:
            accumulator = Accumulator(int_bits, frac_bits)
            usage += accumulator.usage
            int_bits, frac_bits = accumulator.out_bits
            if int_bits > 3:
                int_bits = 3
            divider = Divider(int_bits, frac_bits)
            usage += divider.usage
            cycle += divider.cycle * math.ceil(out_features / act_tile_size)
            int_bits, frac_bits = divider.out_bits
        self.usage = usage
        self.cycle = cycle
        self.out_bits = (int_bits, frac_bits)


class FPGAModel():

    def __init__(self, data, gnn_graph):
        layers = []
        for i, layer in enumerate(gnn_graph):
            layers.append(
                Layer(data, **layer)
                )
        self.usage = sum([layer.usage for layer in layers])
        self.cycle = sum([layer.cycle for layer in layers])

    def validate(self, rResource=None, rCycle=1e5):
        if rResource is None:
            rResource = get_fpga_capacity()
        # rLUT, rReg, rDSP = rResource.LUT, rResource.Reg, rResource.DSP
        # metric = 0
        # for r in ["LUT", "Reg", "DSP"]:
        #     required = getattr(rResource, r)
        #     usage = getattr(self.usage, r)
            # score = (required - usage) / required if usage > required else 0
            # if usage == 0:
            #     score = 0
            # else:
            #     score = math.log(required / usage)
            # score = 1 - (usage / required)
            # metric += 0.25 * score
        # score = (rCycle - self.cycle) / rCycle if self.cycle > rCycle \
            # else 0
        # score = math.log(rCycle / self.cycle)
        # score = 1 - (self.cycle / rCycle)
        # metric += 0.25 * score
        return (self.usage < rResource) and (self.cycle < rCycle)
        # return metric


if __name__ == "__main__":
    import utils
    from dataset import cora
    from .fpga import load_fpga_info

    fpga_info = utils.from_json("./json/fpga/fpga_info.json")
    load_fpga_info(fpga_info)
    gnn_graph = utils.from_json("./json/sample.json")
    fpga_model = FPGAModel(cora[0], gnn_graph)
    print(fpga_model.usage)
    print(fpga_model.cycle)
    print(fpga_model.validate())
