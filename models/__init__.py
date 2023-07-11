from .networks import BottleneckMLP, StandardMLP


def get_architecture(
    architecture="B_6-Wi_1024",
    model="BottleneckMLP",
    num_channels=3,
    resolution=None,
    num_classes=None,
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

        return BottleneckMLP(
            dim_in=resolution**2 * num_channels,
            dim_out=num_classes,
            block_dims=blocks,
        )
    elif model == "MLP":
        blocks = [thin for _ in range(num_blocks)]

        return StandardMLP(
            dim_in=resolution.res**2 * num_channels,
            dim_out=num_classes,
            widths=blocks,
        )
