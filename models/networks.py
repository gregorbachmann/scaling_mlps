from torch import nn


NORMS = {"layer": nn.LayerNorm, "batch": nn.BatchNorm1d}


def get_projector(dim, projector):
    if projector is None:
        return None

    modules = nn.ModuleList()

    projector_layers = projector.split("-")
    projector_layers = [dim] + [int(i) for i in projector_layers]

    for i in range(len(projector_layers) - 1):
        modules.append(
            nn.Linear(int(projector_layers[i]), int(projector_layers[i + 1]))
        )
        modules.append(nn.GELU())

    return nn.Sequential(*modules)


class StandardMLP(nn.Module):
    def __init__(self, dim_in, dim_out, widths, projector=None):
        super(StandardMLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.widths = widths
        self.linear_in = nn.Linear(self.dim_in, self.widths[0])
        self.linear_out = nn.Linear(self.widths[-1], self.dim_out)
        self.projector = get_projector(self.widths[-1], projector)
        self.layers = []
        self.layer_norms = []
        for i in range(len(self.widths) - 1):
            self.layers.append(nn.Linear(self.widths[i], self.widths[i + 1]))
            self.layer_norms.append(nn.LayerNorm(widths[i + 1]))

        self.layers = nn.ModuleList(self.layers)
        self.layernorms = nn.ModuleList(self.layer_norms)

    def forward(self, x, pre_logits=False):
        x = self.linear_in(x)
        for layer, norm in zip(self.layers, self.layer_norms):
            x = norm(x)
            x = nn.GELU()(x)
            x = layer(x)

        if pre_logits:
            return x

        out = self.linear_out(x)

        if self.projector is not None:
            proj = self.projector(x)
            return out, proj

        return out


class BottleneckMLP(nn.Module):
    def __init__(self, dim_in, dim_out, block_dims, norm="layer", projector=None):
        super(BottleneckMLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.block_dims = block_dims
        self.norm = NORMS[norm]

        self.linear_in = nn.Linear(self.dim_in, self.block_dims[0][1])
        self.linear_out = nn.Linear(self.block_dims[-1][1], self.dim_out)
        self.projector = get_projector(self.block_dims[-1][1], projector)
        blocks = []
        layernorms = []

        for block_dim in self.block_dims:
            wide, thin = block_dim
            blocks.append(BottleneckBlock(thin=thin, wide=wide))
            layernorms.append(self.norm(thin))

        self.blocks = nn.ModuleList(blocks)
        self.layernorms = nn.ModuleList(layernorms)

    def forward(self, x, pre_logits=False):
        x = self.linear_in(x)

        for block, norm in zip(self.blocks, self.layernorms):
            x = x + block(norm(x))

        if pre_logits:
            return x

        out = self.linear_out(x)

        if self.projector is not None:
            proj = self.projector(x)
            return out, proj

        return out


class BottleneckBlock(nn.Module):
    def __init__(self, thin, wide):
        super(BottleneckBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(thin, wide), nn.GELU(), nn.Linear(wide, thin)
        )

    def forward(self, x):
        out = self.block(x)

        return out
