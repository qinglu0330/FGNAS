import math
from . import fpga
from .fpga import Resource


def trunc(index, max_bit):
    res = []
    for idx, m in zip(index, max_bit):
        res.append(min(idx, m))
    return res


class Operator():

    def __init__(self, op_type, **bit_width):
        self.type = op_type
        index = list(bit_width.values())
        self.profiling(index)

    def profiling(self, index=None):
        if index is None or len(index) == 0:
            self.usage = Resource()
            self.cycle = 0
            return
        usage = Resource()
        table = fpga.get_operator_info(self.type)
        if len(index) == 1:
            index = trunc(index, [9])
            try:
                info = table[index[0]]
            except IndexError:
                raise IndexError(f"index error, {self.type}, index={index}")
        else:
            index = trunc(index, [3, 6])
            try:
                info = table[index[0]][index[1]]
            except KeyError:
                raise KeyError(f"index error, {self.type}, index={index}")
            except IndexError:
                raise IndexError(f"index error, {self.type}, index={index}")

        for k, v in info.items():
            if hasattr(usage, k):
                v = v or 0
                setattr(usage, k, v)
            else:
                setattr(self, k, v)
        self.usage = usage
        if hasattr(self, 'cycle') is False:
            self.cycle = 0
        return

    def get_usage(self):
        return self.usage

    def get_cycle(self):
        return self.cycle

    def __repr__(self):
        msg = f"Operator {self.type}, usage: {self.usage}, cycle: {self.cycle}"
        return msg


################################################################
# ALUs
################################################################
class Adder(Operator):
    _type = "adder"

    def __init__(self, int_bits, frac_bits):
        if int_bits == 0:
            frac_bits = 0
        bit_width = int_bits + frac_bits
        super().__init__(Adder._type, bit_width=bit_width)
        if int_bits == 0:
            self.out_bits = (0, 0)
        else:
            self.out_bits = (int_bits + 1, frac_bits)


class Multiplier(Operator):
    _type = "multiplier"

    def __init__(self, a_int_bits, a_frac_bits, b_int_bits, b_frac_bits):
        assert (a_int_bits == 0 and b_int_bits == 0) or \
            (a_int_bits != 0 and b_int_bits != 0)
        if a_int_bits == 0:
            a_frac_bits = 0
        if b_int_bits == 0:
            b_frac_bits = 0
        a_bits = a_int_bits + a_frac_bits
        b_bits = b_int_bits + b_frac_bits
        super().__init__(Multiplier._type, a_bits=a_bits, b_bits=b_bits)
        if a_int_bits == 0:
            self.out_bits = (0, 0)
        else:
            self.out_bits = (a_int_bits + b_int_bits - 1,
                             a_frac_bits + b_frac_bits)


class Divider(Operator):
    _type = "divider"

    def __init__(self, int_bits, frac_bits):
        super().__init__(Divider._type, int_bits=int_bits, frac_bits=frac_bits)
        self.out_bits = (int_bits, frac_bits)


class MAX_2(Operator):
    _type = "max_2"

    def __init__(self, int_bits, frac_bits):
        if int_bits == 0:
            frac_bits = 0
        bit_width = int_bits + frac_bits
        super().__init__(MAX_2._type, bit_width=bit_width)
        self.out_bits = (int_bits, frac_bits)


class Truncator_1(Operator):
    _type = "truncator_1"

    def __init__(self, int_bits, frac_bits):
        bit_width = int_bits + frac_bits
        super().__init__(Truncator_1._type, bit_width=bit_width)
        self.cycle = 0
        self.out_bits = (int_bits, frac_bits)


################################################################
# Routing
################################################################
class MUX_2(Operator):
    _type = "mux_2"

    def __init__(self, int_bits, frac_bits):
        bit_width = int_bits + frac_bits
        super().__init__(MUX_2._type, bit_width=bit_width)
        self.out_bits = (int_bits, frac_bits)


################################################################
# Activation
################################################################
class ReLU(Operator):
    _type = "relu"

    def __init__(self, int_bits, frac_bits):
        bit_width = int_bits + frac_bits
        super().__init__(ReLU._type, bit_width=bit_width)
        self.out_bits = (int_bits, frac_bits)


class ELU(Operator):
    _type = "elu"

    def __init__(self, int_bits, frac_bits):
        super().__init__(ELU._type, int_bits=int_bits, frac_bits=frac_bits)
        self.out_bits = (int_bits, frac_bits)


class Tanh(Operator):
    _type = "tanh"

    def __init__(self, int_bits, frac_bits):
        super().__init__(Tanh._type, int_bits=int_bits, frac_bits=frac_bits)
        self.out_bits = (int_bits, frac_bits)


class Sigmoid(Operator):
    _type = "sigmoid"

    def __init__(self, int_bits, frac_bits):
        super().__init__(Sigmoid._type, int_bits=int_bits, frac_bits=frac_bits)
        self.out_bits = (int_bits, frac_bits)


class Identity(Operator):
    _type = "identity"

    def __init__(self, *args, **kwargs):
        super().__init__(Identity._type)

    def __repr__(self):
        msg = "Identity ()"
        return msg


################################################################
# Complex Operators
################################################################
class Complex():

    def __init__(self, type_, components, cycle):
        self.type = type_
        self.components = components
        self.cycle = cycle
        self.profiling()

    def profiling(self):
        # print(self.components)
        if self.components == []:
            self.usage = Resource()
        else:
            self.usage = sum([c.usage for c in self.components])

    def __repr__(self):
        msg = (f"Complex Operator: {self.type}, Usage: {self.usage}, "
               f"Cycle: {self.cycle}")
        return msg


class MulArray(Complex):
    _type = "mul_array"

    def __init__(self, width, a_int_bits, a_frac_bits, b_int_bits,
                 b_frac_bits):
        operator = Multiplier(a_int_bits, a_frac_bits, b_int_bits, b_frac_bits)
        components = [operator] * width
        cycle = operator.cycle
        super().__init__(MulArray._type, components, cycle)
        self.width = width
        self.out_bits = (a_int_bits, a_frac_bits)


class AdderTree(Complex):
    _type = "adder_tree"

    def __init__(self, num_inputs, int_bits, frac_bits):
        components = []
        cycle = 0
        while num_inputs > 1:
            num_adders = math.floor(num_inputs/2)
            operator = Adder(int_bits, frac_bits)
            cycle += operator.cycle
            components += [operator] * num_adders
            if int_bits != 0:
                int_bits += 1
            num_inputs = math.ceil(num_inputs/2)
        super().__init__(MAC._type, components, cycle)
        self.out_bits = (int_bits, frac_bits)


class MulAdderTree(Complex):
    _type = "mul_adder_tree"

    def __init__(self, num_input_pairs, act_int_bits, act_frac_bits,
                 weight_int_bits, weight_frac_bits):
        components = []
        cycle = 0
        mul_array = MulArray(num_input_pairs, act_int_bits, act_frac_bits,
                             weight_int_bits, weight_frac_bits)
        components.append(mul_array)
        cycle += mul_array.cycle
        int_bits, frac_bits = mul_array.out_bits

        adder_tree = AdderTree(mul_array.width, int_bits, frac_bits)
        components.append(adder_tree)
        cycle += adder_tree.cycle
        super().__init__(MulAdderTree._type, components, cycle)
        self.out_bits = (int_bits, frac_bits)


class MulAdderTreeArray(Complex):
    _type = "mul_adder_array"

    def __init__(self, width, num_input_pairs, act_int_bits, act_frac_bits,
                 weight_int_bits, weight_frac_bits):
        components = []
        cycle = 0
        operator = MulAdderTree(num_input_pairs, act_int_bits, act_frac_bits,
                                weight_int_bits, weight_frac_bits)
        components += [operator] * width
        cycle += operator.cycle
        super().__init__(MulAdderTreeArray._type, components, cycle)
        self.out_bits = operator.out_bits
        self.width = width


class Accumulator(Complex):
    _type = "accumulator"

    def __init__(self, int_bits, frac_bits):
        components = []
        cycle = 0
        adder = Adder(int_bits, frac_bits)
        components.append(adder)
        cycle += adder.cycle
        int_bits, frac_bits = adder.out_bits

        register = Register(1, int_bits + frac_bits)
        components.append(register)

        if int_bits > 0:
            truncator = Truncator_1(int_bits, frac_bits)
            components.append(truncator)
            cycle += truncator.cycle
        super().__init__(Accumulator._type, components, cycle)
        self.out_bits = (int_bits, frac_bits)


class MaxAccumulator(Complex):
    _type = "max_accumulator"

    def __init__(self, int_bits, frac_bits):
        components = []
        cycle = 0
        max_ = MAX_2(int_bits, frac_bits)
        components.append(max_)
        cycle += max_.cycle

        int_bits, frac_bits = max_.out_bits
        register = Register(1, int_bits + frac_bits)
        components.append(register)
        super().__init__(MaxAccumulator._type, components, cycle)
        self.out_bits = (int_bits, frac_bits)


class MAC(Complex):
    _type = "mac"

    def __init__(self, act_int_bits, act_frac_bits, weight_int_bits,
                 weight_frac_bits):
        components = []
        cycle = 0
        multiplier = Multiplier(
            act_int_bits, act_frac_bits, weight_int_bits, weight_frac_bits)
        components.append(multiplier)
        cycle += multiplier.cycle
        int_bits, frac_bits = multiplier.out_bits

        if int_bits == 0:
            product_bits = 0
            out_bits = (0, 0)
        else:
            product_bits = int_bits + frac_bits
            out_bits = (int_bits + 1, frac_bits)
        adder = Adder(int_bits, frac_bits)
        components.append(adder)
        cycle += adder.cycle
        if product_bits > 0:
            truncator = Truncator_1(*out_bits)
            components.append(truncator)
            cycle += truncator.cycle
        register = Register(1, sum(out_bits))
        components.append(register)
        super().__init__(MAC._type, components, cycle)
        self.out_bits = out_bits


class MACArray(Complex):
    _type = "mac_array"

    def __init__(self, width, act_int_bits, act_frac_bits, weight_int_bits,
                 weight_frac_bits):
        components = []
        cycle = []
        operator = MAC(
            act_int_bits, act_frac_bits, weight_int_bits, weight_frac_bits)
        components += [operator] * width
        cycle += operator.cycle
        super().__init__(MACArray._type, components, cycle)
        self.width = width
        self.out_bits = operator.out_bits


class GenericMAC(Complex):
    _type = "generic_mac"

    def __init__(self, num_input_pairs, act_int_bits, act_frac_bits,
                 weight_int_bits, weight_frac_bits):
        components = []
        cycle = 0
        mul_array = MulArray(num_input_pairs, act_int_bits, act_frac_bits,
                             weight_int_bits, weight_frac_bits)
        components.append(mul_array)
        cycle += mul_array.cycle

        adder_tree = AdderTree(mul_array.width, *mul_array.out_bits)
        components.append(adder_tree)
        cycle += adder_tree.cycle

        int_bits, frac_bits = mul_array.out_bits
        if int_bits == 0:
            sum_bits = 0
            out_bits = (0, 0)
        else:
            sum_bits = int_bits + frac_bits
            out_bits = (int_bits + 1, frac_bits)
        adder = Adder(int_bits, frac_bits)
        components.append(adder)
        cycle += adder.cycle
        if sum_bits > 0:
            truncator = Truncator_1(int_bits, frac_bits)
            components.append(truncator)
            cycle += truncator.cycle
        register = Register(1, sum(out_bits))
        components.append(register)
        # print(components)
        super().__init__(GenericMAC._type, components, cycle)
        self.out_bits = out_bits


class GenericMACArray(Complex):
    _type = "generic mac_array"

    def __init__(self, width, num_input_pairs, act_int_bits, act_frac_bits,
                 weight_int_bits, weight_frac_bits):
        # print(act_int_bits, act_frac_bits, weight_int_bits, weight_frac_bits)
        components = []
        cycle = 0
        operator = GenericMAC(num_input_pairs, act_int_bits, act_frac_bits,
                              weight_int_bits, weight_frac_bits)
        components += [operator] * width
        cycle += operator.cycle
        super().__init__(GenericMACArray._type, components, cycle)
        self.width = width
        self.out_bits = operator.out_bits


class Register():
    _type = "register"

    def __init__(self, width, bit_width=0):
        self.width = width
        self.bit_width = bit_width
        self.usage = Resource(0, width * bit_width, 0)
        self.cycle = 0

    def __repr__(self):
        msg = f"Register-{self.width} ({self.bit_width})"
        return msg
