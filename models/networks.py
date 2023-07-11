from torch import nn


class StandardMLP(nn.Module):
    def __init__(self, dim_in, dim_out, widths):
        super(StandardMLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.widths = widths
        self.linear_in = nn.Linear(self.dim_in, self.widths[0])
        self.linear_out = nn.Linear(self.widths[-1], self.dim_out)
        self.layers = []
        self.layer_norms = []
        for i in range(len(self.widths) - 1):
            self.layers.append(nn.Linear(self.widths[i], self.widths[i + 1]))
            self.layer_norms.append(nn.LayerNorm(widths[i + 1]))

        self.layers = nn.ModuleList(self.layers)
        self.layernorms = nn.ModuleList(self.layer_norms)

    def forward(self, x):
        z = self.linear_in(x)
        for layer, norm in zip(self.layers, self.layer_norms):
            z = norm(z)
            z = layer(z)

        out = self.linear_out(z)

        return out


class BottleneckMLP(nn.Module):
    def __init__(self, dim_in, dim_out, block_dims):
        super(BottleneckMLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.block_dims = block_dims

        self.linear_in = nn.Linear(self.dim_in, self.block_dims[0][1])
        self.linear_out = nn.Linear(self.block_dims[-1][1], self.dim_out)
        blocks = []
        layernorms = []

        for block_dim in self.block_dims:
            wide, thin = block_dim
            blocks.append(BottleneckBlock(thin=thin, wide=wide))
            layernorms.append(nn.LayerNorm(thin))

        self.blocks = nn.ModuleList(blocks)
        self.layernorms = nn.ModuleList(layernorms)

    def forward(self, x):
        x = self.linear_in(x)

        for block, norm in zip(self.blocks, self.layernorms):
            x = x + block(norm(x))

        out = self.linear_out(x)

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


class Linear(nn.Module):
    """For readability"""

    def __init__(self, dim_in, dim_out):
        super(Linear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.linear = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, x):
        return self.linear(x)
