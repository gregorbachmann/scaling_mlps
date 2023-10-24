import os
import json


def config_to_name(args):
    return os.path.join(
        str(args.dataset) + '_res_' + str(args.crop_resolution),
        f"{args.model}_{args.architecture}_norm_{args.normalization}",
        f"batchsize_{args.batch_size}",
        f"{args.optimizer}_lr_{args.lr}_smooth_{args.smooth}"
        + f"_decay_{args.weight_decay}_augment_{args.augment}_mixup_{args.mixup}_droprate_{args.drop_rate}",
        f"ntrain_{args.n_train}",
    )


def model_from_config(path):
    """Return model class from checkpoint path."""
    path = os.path.dirname(path)
    with open(path + '/config.txt', 'r') as f:
        config = json.load(f)
    model = config["model"]
    architecture = config["architecture"]
    norm = config["normalization"]
    crop_resolution = int(config["crop_resolution"])

    return model, architecture, crop_resolution, norm
