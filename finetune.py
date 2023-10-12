import argparse
import os
import json

import torch
from torch.nn import CrossEntropyLoss, Linear
from tqdm import tqdm

from data_utils.data_stats import *
from data_utils.dataloader import get_loader
from models import get_architecture
from utils.config import config_to_name, model_from_config
from utils.metrics import topk_acc, real_acc
from utils.optimizer import (
    OPTIMIZERS_DICT,
    SCHEDULERS,
    get_optimizer,
    get_scheduler,
)
from train import train, test


@torch.no_grad()
def test_time_aug(model, loader, num_augs, args):
    model.eval()
    all_preds = torch.zeros(len(loader.indices), model.linear_out.out_features)

    for _ in tqdm(range(num_augs)):
        targets = []
        cnt = 0

        for ims, targs in loader:
            ims = torch.reshape(ims, (ims.shape[0], -1))
            preds = model(ims)

            all_preds[cnt:cnt + ims.shape[0]] += torch.nn.functional.softmax(preds.detach().cpu(), dim=-1)
            targets.append(targs.detach().cpu())

            cnt += ims.shape[0]

    all_preds = all_preds / num_augs
    targets = torch.cat(targets)

    if args.dataset != 'imagenet_real':
        acc, top5 = topk_acc(all_preds, targets, k=5, avg=True)
    else:
        acc = real_acc(all_preds, targets, k=5, avg=True)
        top5 = 0.

    return acc, top5


def finetune(args):
    # Use mixed precision matrix multiplication
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Get the resolution the model was trained with
    model, architecture, crop_resolution, norm = model_from_config(args.checkpoint_path)
    args.model = model
    args.architecture = architecture
    args.crop_resolution = crop_resolution
    args.normalization = norm
    model = get_architecture(**args.__dict__).cuda()

    # Get the dataloaders
    train_loader = get_loader(
        args.dataset,
        bs=args.batch_size,
        mode='train',
        augment=args.augment,
        dev=device,
        num_samples=args.n_train,
        mixup=args.mixup,
        data_path=args.data_path,
        data_resolution=args.data_resolution,
        crop_resolution=args.crop_resolution,
        crop_ratio=tuple(args.crop_ratio),
        crop_scale=tuple(args.crop_scale)
    )

    test_loader = get_loader(
        args.dataset,
        bs=args.batch_size,
        mode='test',
        augment=False,
        dev=device,
        data_path=args.data_path,
        data_resolution=args.data_resolution,
        crop_resolution=args.crop_resolution,
    )

    test_loader_aug = get_loader(
        args.dataset,
        bs=args.batch_size,
        mode='test',
        augment=True,
        dev=device,
        data_path=args.data_path,
        data_resolution=args.data_resolution,
        crop_resolution=args.crop_resolution,
        crop_ratio=tuple(args.crop_ratio),
        crop_scale=tuple(args.crop_scale)

    )

    # Create unique identifier
    name = config_to_name(args)
    path = os.path.join(args.checkpoint_folder, name)

    # Create folder to store the checkpoints
    if not os.path.exists(path):
        os.makedirs(path)
        with open(path + '/config.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    print('Loading checkpoint', args.checkpoint_path)
    params = {
        k: v
        for k, v in torch.load(args.checkpoint_path).items()
        if 'linear_out' not in k
    }

    print('Load_state output', model.load_state_dict(params, strict=False))

    model.linear_out = Linear(model.linear_out.in_features, args.num_classes)
    model.cuda()

    param_groups = [
        {
            'params': [v for k, v in model.named_parameters() if 'linear_out' in k],
            'lr': args.lr,
        },
    ]

    if args.mode != "linear":
        param_groups.append(
            {
                'params': [
                    v for k, v in model.named_parameters() if 'linear_out' not in k
                ],
                'lr': args.lr * args.body_learning_rate_multiplier,
            },
        )
    else:
        # freeze the body
        for name, param in model.named_parameters():
            if 'linear_out' not in name:
                param.requires_grad = False

    opt = get_optimizer(args.optimizer)(param_groups, lr=args.lr)

    scheduler = get_scheduler(opt, args.scheduler, **args.__dict__)
    loss_fn = CrossEntropyLoss(label_smoothing=args.smooth)

    for ep in range(args.epochs):
        train_acc, train_top5, train_loss, train_time = train(
            model, opt, scheduler, loss_fn, ep, train_loader, args
        )

        if (ep + 1) % args.calculate_stats == 0:
            test_acc, test_top5, test_loss, test_time = test(
                model, test_loader, loss_fn, args
            )

            # Print all the stats
            print('Epoch', ep, '       Time:', train_time)
            print('-------------- Training ----------------')
            print('Average Training Loss:       ', '{:.6f}'.format(train_loss))
            print('Average Training Accuracy:   ', '{:.4f}'.format(train_acc))
            print('Top 5 Training Accuracy:     ', '{:.4f}'.format(train_top5))
            print('---------------- Test ------------------')
            print('Test Accuracy        ', '{:.4f}'.format(test_acc))
            print('Top 5 Test Accuracy          ', '{:.4f}'.format(test_top5))
            print()

    print('-------- Test Time Augmentation Evaluation -------')
    
    num_augs = 100
    acc, top5 = test_time_aug(model, test_loader_aug, num_augs, args)
    print(num_augs, 'augmentations: Test accuracy:', acc)
    print(num_augs, 'augmentations: Test Top5 accuracy:', top5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scaling MLPs")
    # Data
    parser.add_argument(
        "--data_path", default="./beton", type=str, help="Path to data directory"
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
        "--checkpoint_path", default=None, type=str, help="Checkpoint", required=True
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
    args = parser.parse_args()

    args.num_classes = CLASS_DICT[args.dataset]

    if args.n_train is None:
        args.n_train = SAMPLE_DICT[args.dataset]

    finetune(args)
