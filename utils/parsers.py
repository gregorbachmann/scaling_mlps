import argparse
from utils.optimizer import OPTIMIZERS_DICT, SCHEDULERS


def get_training_parser():
    parser = argparse.ArgumentParser(description="Scaling MLPs")

    # Data
    parser.add_argument(
        "--data_path", 
        default="./beton", 
        type=str,
        help="Path to data directory"
    )
    parser.add_argument(
        "--dataset", 
        default="imagenet21", 
        type=str, 
        help="Dataset"
    )
    parser.add_argument(
        "--resolution", 
        default=64, 
        type=int, 
        help="Image Resolution"
    )
    parser.add_argument(
        "--crop_resolution", 
        default=None, 
        type=int, 
        help="Crop Resolution"
    )
    parser.add_argument(
        "--n_train", 
        default=None, 
        type=int, 
        help="Number of samples. None for all"
    )

    # Model
    parser.add_argument(
        "--model", 
        default="BottleneckMLP", 
        type=str, 
        help="Type of model"
    )
    parser.add_argument(
        "--architecture", 
        default="B_6-Wi_1024", 
        type=str, 
        help="Architecture type"
    )
    parser.add_argument(
        "--normalization", 
        default="layer", 
        type=str, 
        help="Normalization type"
    )
    parser.add_argument(
        "--act", 
        default="gelu", 
        type=str, 
        help="Normalization type"
    )
    parser.add_argument(
        "--drop_rate", 
        default=None, 
        type=float, 
        help="Drop rate for dropout"
    )

    # Training
    parser.add_argument(
        "--optimizer",
        default="lion",
        type=str,
        help="Choice of optimizer",
        choices=OPTIMIZERS_DICT.keys(),
    )
    parser.add_argument(
        "--batch_size", 
        default=4096, 
        type=int, 
        help="Batch size"
    )
    parser.add_argument(
        "--accum_steps", 
        default=1, 
        type=int, 
        help="Number of accumulation steps"
    )
    parser.add_argument(
        "--lr", 
        default=0.00005, 
        type=float, 
        help="Learning rate"
    )
    parser.add_argument(
        "--scheduler", 
        type=str, 
        default="none", 
        choices=SCHEDULERS, 
        help="Scheduler"
    )
    parser.add_argument(
        "--weight_decay", 
        default=0.0, 
        type=float, 
        help="Weight decay"
    )
    parser.add_argument(
        "--epochs", 
        default=500, 
        type=int, 
        help="Epochs"
    )
    parser.add_argument(
        "--smooth", 
        default=0.3, 
        type=float, 
        help="Amount of label smoothing"
    )
    parser.add_argument(
        "--clip", 
        default=0., 
        type=float, 
        help="Gradient clipping"
        )
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reinitialize from checkpoint",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to augment data",
    )
    parser.add_argument(
        "--mixup", 
        default=0.8, 
        type=float, 
        help="Strength of mixup"
        )
    
    # Logging
    parser.add_argument(
        "--calculate_stats",
        type=int,
        default=1,
        help="Frequence of calculating stats",
    )
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default="./checkpoints",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--save_freq", 
        type=int,
        default=50, 
        help="Save frequency"
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save checkpoints",
    )
    parser.add_argument(
        "--wandb",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to log with wandb",
    )
    parser.add_argument(
        "--wandb_project", 
        default="mlps", 
        type=str, 
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_entity",
        default=None, 
        type=str, 
        help="Wandb entity name"
    )

    return parser


def get_finetune_parser():
    parser = argparse.ArgumentParser(description="Scaling MLPs")
    # Data
    parser.add_argument(
        "--data_path", default="./beton", type=str, help="Path to data directory"
    )
    parser.add_argument(
        "--architecture", default="B_12-Wi_1024", type=str, help="Path to data directory"
    )
    parser.add_argument("--dataset", default="cifar100", type=str, help="Dataset")
    parser.add_argument("--data_resolution", default=64, type=int, help="Image Resolution")
    parser.add_argument(
        "--n_train", default=None, type=int, help="Number of samples. None for all"
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to augment data",
    )
    parser.add_argument("--mixup", default=0., type=float, help="Strength of mixup")
    parser.add_argument('--crop_scale', nargs='+', type=float, default=[0.08, 1.], help="Scale for crop at test time")
    parser.add_argument('--crop_ratio', nargs='+', type=float, default=[0.08, 1.], help="Ratio for crop at test time")
    parser.add_argument(
        "--drop_rate", 
        default=None, 
        type=float, 
        help="Drop rate for dropout"
    )

    # Training
    parser.add_argument(
        "--optimizer",
        default="sgd",
        type=str,
        help="Choice of optimizer",
        choices=OPTIMIZERS_DICT.keys(),
    )
    parser.add_argument("--batch_size", default=4096, type=int, help="Batch size")
    parser.add_argument("--accum_steps", default=1, type=int, help="Number of accumulation steps")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    parser.add_argument(
        "--scheduler", type=str, default="none", choices=SCHEDULERS, help="Scheduler"
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay")
    parser.add_argument("--epochs", default=500, type=int, help="Epochs")
    parser.add_argument(
        "--smooth", default=0.3, type=float, help="Amount of label smoothing"
    )
    parser.add_argument("--clip", default=1.0, type=float, help="Gradient clipping")

    # Misc
    parser.add_argument(
        "--mode",
        default="linear",
        type=str,
        help="Mode",
        choices=["linear", "finetune"],
    )
    parser.add_argument(
        "--checkpoint_folder",
        default="./checkpoints_finetune",
        type=str,
        help="Folder to store checkpoints",
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Checkpoint", required=False
    )
    parser.add_argument(
        "--checkpoint", default='checkpoints_finetune', type=str, help="Checkpoint", required=False
    )
    parser.add_argument(
        "--body_learning_rate_multiplier",
        default=0.1,
        type=float,
        help="Percentage of learning rate for the body",
    )
    parser.add_argument(
        "--calculate_stats",
        type=int,
        default=1,
        help="Frequency of calculating stats",
    )
    parser.add_argument(
        "--save_freq", 
        type=int,
        default=20, 
        help="Save frequency"
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save checkpoints",
    )

    return parser