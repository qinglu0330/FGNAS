class Resource():

    def __init__(self, LUT=0, Reg=0, DSP=0):
        self.LUT = LUT
        self.Reg = Reg
        self.DSP = DSP

    def __repr__(self):
        msg = f"Resource: {self.LUT} LUTs, {self.Reg} Registers, " + \
              f"{self.DSP} DSPs"
        return msg

    def duplicate(self, repetition):
        self.LUT *= repetition
        self.Reg *= repetition
        self.DSP *= repetition
        return self

    def __add__(self, other):
        # if other.LUT is None:
        #     LUT = self.LUT
        # else:
        #     LUT = self.LUT + other.LUT
        # if other.Reg is None:
        #     Reg = self.Reg
        # else:
        #     Reg = self.Reg + other.Reg
        # if other.DSP is None:
        #     DSP = self.DSP
        # else:
        #     DSP = self.DSP + other.DSP
        LUT = self.LUT + other.LUT
        Reg = self.Reg + other.Reg
        DSP = self.DSP + other.DSP
        return Resource(LUT, Reg, DSP)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __mul__(self, n):
        LUT = self.LUT * n
        Reg = self.Reg * n
        DSP = self.DSP * n
        return Resource(LUT, Reg, DSP)

    def __lt__(self, other):
        if self.LUT < other.LUT:
            return True
        if self.Reg < other.Reg:
            return True
        if self.DSP < other.DSP:
            return True
        return False


class FPGA():

    def __init__(self, family, capacity):
        self.capacity = capacity
        self.family = family
        self.ops = {}

    def __repr__(self):
        return f"FPGA: {self.family}, {str(self.capacity)}"

    def set_operator_info(self, op_type: str, table):
        setattr(self, op_type, table)

    def get_operator_info(self, op_type: str):
        return self.ops[op_type]

    def load_info(self, **info):
        for op, table in info.items():
            setattr(self, op, table)

    def load_operator(self, op_type, table):
        setattr(self, op_type, table)
        self.ops[op_type] = table


fpga = FPGA("default", Resource(1000000, 10000, 100))


def load_fpga_info(info):
    global fpga
    for op, table in info.items():
        fpga.load_operator(op, table)
    return


def get_operator_info(op_type: str):
    return fpga.get_operator_info(op_type)


def get_fpga_capacity():
    return fpga.capacity


def get_cycle():
    return fpga.cycle


def show_base():
    for k, v in fpga.ops.items():
        print(k)


if __name__ == "__main__":
    import utils

    fpga_info = utils.from_json("./json/fpga/fpga_info.json")
    load_fpga_info(fpga_info)
    print(fpga.ops)
