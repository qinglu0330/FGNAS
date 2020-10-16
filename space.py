ARCH_SPACE = {
    "attention": ("constant", "gcn", "gat"),
    "heads": (1, 2, 4, 8, 16),
    "aggregation": ("add", "mean", "max"),
    # "combine_function": ("none", "concat"),
    "activation": ("relu", "tanh", "sigmoid", "elu"),
    "hidden_features": (4, 8, 16, 32, 64),
}

QUANT_SPACE = {
    "act_int_bits": (1, 2, 3),
    "act_frac_bits": (0, 1, 2, 3, 4, 5, 6),
    "weight_int_bits": (1, 2, 3),
    "weight_frac_bits": (0, 1, 2, 3, 4, 5, 6)
}

HW_SPACE = {
    "in_tile_size": (1, 2, 4, 8, 16),
    "out_tile_size": (1, 2, 4, 8, 16),
    "aggr_tile_size": (1, 2, 4, 8, 16),
    "attn_tile_size": (1, 2, 4, 8, 16),
    "act_tile_size": (1, 2, 4, 8, 16)
}
