from .networks import BottleneckMLP, StandardMLP


def get_architecture(
    architecture="B_6-Wi_1024",
    model="BottleneckMLP",
    num_channels=3,
    crop_resolution=None,
    num_classes=None,
    normalization='layer',
    act='gelu',
    drop_rate=None,
    **kwargs,
):
    assert model in ["BottleneckMLP", "MLP"], f"Model {model} not supported."

    sep = architecture.split("-")
    num_blocks = int(sep[0].split("_")[1])
    thin = int(sep[1].split("_")[1])

    if len(sep) == 3:
        expansion_factor = int(sep[2].split("_")[1])
    else:
        expansion_factor = 4

    if model == "BottleneckMLP":
        blocks = [[expansion_factor * thin, thin] for _ in range(num_blocks)]
        dim_in = crop_resolution**2 * num_channels
        return BottleneckMLP(
            dim_in=dim_in,
            dim_out=num_classes,
            block_dims=blocks,
            norm=normalization,
            act=act,
            drop_rate=drop_rate
        )

    elif model == "MLP":
        blocks = [thin for _ in range(num_blocks)]

        return StandardMLP(
            dim_in=crop_resolution**2 * num_channels,
            dim_out=num_classes,
            widths=blocks,
        )
