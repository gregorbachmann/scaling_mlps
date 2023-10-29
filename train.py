import os
import time
import json

import torch
import wandb
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from models import get_architecture
from data_utils.data_stats import *
from data_utils.dataloader import get_loader
from utils.config import config_to_name
from utils.get_compute import get_compute
from utils.metrics import topk_acc, real_acc, AverageMeter
from utils.optimizer import get_optimizer, get_scheduler, OPTIMIZERS_DICT, SCHEDULERS
from utils.parsers import get_training_parser


def train(model, opt, scheduler, loss_fn, epoch, train_loader, args):
    start = time.time()
    model.train()

    total_acc, total_top5 = AverageMeter(), AverageMeter()
    total_loss = AverageMeter()

    for step, (ims, targs) in enumerate(tqdm(train_loader, desc="Training epoch: " + str(epoch))):
        ims = torch.reshape(ims, (ims.shape[0], -1))
        preds = model(ims)

        if args.mixup > 0:
            targs_perm = targs[:, 1].long()
            weight = targs[0, 2].squeeze()
            targs = targs[:, 0].long()
            if weight != -1:
                loss = loss_fn(preds, targs) * weight + loss_fn(preds, targs_perm) * (
                    1 - weight
                )
            else:
                loss = loss_fn(preds, targs)
                targs_perm = None
        else:
            loss = loss_fn(preds, targs)
            targs_perm = None

        acc, top5 = topk_acc(preds, targs, targs_perm, k=5, avg=True)
        total_acc.update(acc, ims.shape[0])
        total_top5.update(top5, ims.shape[0])

        loss = loss / args.accum_steps
        loss.backward()
        
        if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            opt.zero_grad()

        total_loss.update(loss.item() * args.accum_steps, ims.shape[0])

    end = time.time()

    scheduler.step()

    return (
        total_acc.get_avg(percentage=True),
        total_top5.get_avg(percentage=True),
        total_loss.get_avg(percentage=False),
        end - start,
    )


@torch.no_grad()
def test(model, loader, loss_fn, args):
    start = time.time()
    model.eval()
    total_acc, total_top5, total_loss = AverageMeter(), AverageMeter(), AverageMeter()

    for ims, targs in tqdm(loader, desc="Evaluation"):
        ims = torch.reshape(ims, (ims.shape[0], -1))
        preds = model(ims)

        if args.dataset != 'imagenet_real':
            acc, top5 = topk_acc(preds, targs, k=5, avg=True)
            loss = loss_fn(preds, targs).item()
        else:
            acc = real_acc(preds, targs, k=5, avg=True)
            top5 = 0
            loss = 0

        total_acc.update(acc, ims.shape[0])
        total_top5.update(top5, ims.shape[0])
        total_loss.update(loss)

    end = time.time()

    return (
        total_acc.get_avg(percentage=True),
        total_top5.get_avg(percentage=True),
        total_loss.get_avg(percentage=False),
        end - start,
    )


def main(args):
    # Use mixed precision matrix multiplication
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_architecture(**args.__dict__).cuda()

    # Count number of parameters for logging purposes
    args.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create unique identifier
    name = config_to_name(args)
    path = os.path.join(args.checkpoint_folder, name)

    # Create folder to store the checkpoints
    if not os.path.exists(path):
        os.makedirs(path)
        with open(path + '/config.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # Get the dataloaders
    local_batch_size = args.batch_size // args.accum_steps

    train_loader = get_loader(
        args.dataset,
        bs=local_batch_size,
        mode="train",
        augment=args.augment,
        dev=device,
        num_samples=args.n_train,
        mixup=args.mixup,
        data_path=args.data_path,
        data_resolution=args.resolution,
        crop_resolution=args.crop_resolution
    )

    test_loader = get_loader(
        args.dataset,
        bs=local_batch_size,
        mode="test",
        augment=False,
        dev=device,
        data_path=args.data_path,
        data_resolution=args.resolution,
        crop_resolution=args.crop_resolution
    )

    start_ep = 1
    if args.reload:
        try:
            params = torch.load(path + "/name_of_checkpoint")
            model.load_state_dict(params)
            start_ep = 350
        except:
            print("No pretrained model found, training from scratch")

    opt = get_optimizer(args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(opt, args.scheduler, **args.__dict__)

    loss_fn = CrossEntropyLoss(label_smoothing=args.smooth)

    if args.wandb:
        # Add your wandb credentials and project name
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args.__dict__,
            tags=["pretrain", args.dataset],
        )
        wandb.run.name = name

    compute_per_epoch = get_compute(model, args.n_train, args.crop_resolution)

    for ep in range(start_ep, args.epochs):
        calc_stats = (ep + 1) % args.calculate_stats == 0

        current_compute = compute_per_epoch * ep

        train_acc, train_top5, train_loss, train_time = train(
            model, opt, scheduler, loss_fn, ep, train_loader, args
        )

        if args.wandb:
            wandb.log({"Training time": train_time, "Training loss": train_loss})

        if ep % args.save_freq == 0 and args.save:
            torch.save(
                model.state_dict(),
                path + "/epoch_" + str(ep) + "_compute_" + str(current_compute),
            )

        if calc_stats:
            test_acc, test_top5, test_loss, test_time = test(
                model, test_loader, loss_fn, args
            )
            if args.wandb:
                wandb.log(
                    {
                        "Training accuracy": train_acc,
                        "Training Top 5 accuracy": train_top5,
                        "Test accuracy": test_acc,
                        "Test Top 5 accuracy": test_top5,
                        "Test loss": test_loss,
                        "Inference time": test_time,
                    }
                )

            # Print all the stats
            print("Epoch", ep, "       Time:", train_time)
            print("-------------- Training ----------------")
            print("Average Training Loss:       ", "{:.6f}".format(train_loss))
            print("Average Training Accuracy:   ", "{:.4f}".format(train_acc))
            print("Top 5 Training Accuracy:     ", "{:.4f}".format(train_top5))
            print("---------------- Test ------------------")
            print("Test Accuracy        ", "{:.4f}".format(test_acc))
            print("Top 5 Test Accuracy          ", "{:.4f}".format(test_top5))
            print()



if __name__ == "__main__":
    parser = get_training_parser()
    args = parser.parse_args()

    args.num_classes = CLASS_DICT[args.dataset]

    if args.n_train is None:
        args.n_train = SAMPLE_DICT[args.dataset]

    if args.crop_resolution is None:
        args.crop_resolution = args.resolution
    if args.wandb_entity is None:
        print("No wandb entity provided, Continuing without wandb")
        args.wandb = False

    main(args)
