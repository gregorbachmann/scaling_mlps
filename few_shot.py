import gc

import numpy as np
import torch
import yaml
from tqdm import tqdm

from sklearn import linear_model
import argparse

import torch
from tqdm import tqdm

from data_utils.data_stats import *
from data_utils.dataloader import get_loader
from models import get_architecture
from utils.config import model_from_config
from utils.metrics import topk_acc


def create_few_shot_loader_fn(loader, n_few_shot=10, num_classes=1000, batch_size=256):
    # Collect data in memory since we're dealing with a small dataset
    classes_seen = {i: 0 for i in range(num_classes)}

    few_shot_dataset_images = []
    few_shot_dataset_targets = []

    iterator = iter(loader)

    for _ in (pbar := tqdm(range(len(loader)))):
        batch = next(iterator)

        images, targets = batch[0].detach().cpu(), batch[1].detach().cpu()

        for i in range(images.shape[0]):
            if classes_seen[targets[i].item()] < n_few_shot:
                few_shot_dataset_images.append(images[i])
                few_shot_dataset_targets.append(targets[i])

                classes_seen[targets[i].item()] += 1

        solved = sum([v == n_few_shot for v in classes_seen.values()])
        pbar.set_description(f"Few shot classes filled {solved / num_classes:.4f}")
        if solved == num_classes:
            break

    ## ffcv cleaning up
    iterator.close()
    del iterator
    gc.collect()

    few_shot_dataset_images = torch.stack(few_shot_dataset_images)
    few_shot_dataset_targets = torch.stack(few_shot_dataset_targets)

    few_shot_dataset = torch.utils.data.TensorDataset(
        few_shot_dataset_images, few_shot_dataset_targets
    )

    few_shot_loader = torch.utils.data.DataLoader(
        few_shot_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return few_shot_loader


def get_embeddings(model, loader):
    features = []
    labels = []

    for batch in tqdm(loader, desc="Getting embeddings"):
        ims, targets = batch[0], batch[1]
        ims = torch.reshape(ims, (ims.shape[0], -1))

        with torch.no_grad():
            features.append(model(ims.cuda(), pre_logits=True).detach().cpu())
            labels.append(targets.detach().cpu())

    features = torch.cat(features)
    labels = torch.cat(labels)

    return features.numpy(), labels.numpy()


def categorical_to_one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]


def one_hot_to_categorical(labels):
    return np.argmax(labels, axis=-1)


def do_few_shot(model, train_loader, test_loader, regularization):
    train_features, train_labels = get_embeddings(model, train_loader)
    test_faetures, test_labels = get_embeddings(model, test_loader)

    reg = linear_model.Ridge(alpha=regularization, solver="cholesky")

    reg.fit(train_features, categorical_to_one_hot(train_labels, 1000))
    predictions = reg.predict(test_faetures)

    accuracies = topk_acc(
        torch.tensor(predictions), torch.tensor(test_labels), avg=True
    )
    return accuracies


def get_loaders(dataset, data_resolution, n_few_shot, num_classes, kwargs):
    print("Getting few shot loaders")
    # Get the dataloaders
    train_loader = get_loader(
        dataset,
        bs=kwargs["batch_size"],
        mode="train",
        augment=False,
        dev=kwargs["device"],
        data_path=kwargs["data_path"],
        data_resolution=data_resolution,
        crop_resolution=kwargs["crop_resolution"],
        sequential_train_order=True,  # just for reproducibility
    )

    test_loader = get_loader(
        dataset,
        bs=kwargs["batch_size"],
        mode="test",
        augment=False,
        dev=kwargs["device"],
        data_path=kwargs["data_path"],
        data_resolution=data_resolution,
        crop_resolution=kwargs["crop_resolution"],
    )

    few_shot_loader = create_few_shot_loader_fn(
        train_loader, n_few_shot, num_classes, kwargs["batch_size"]
    )

    del train_loader

    return few_shot_loader, test_loader


def few_shot(args, verbose=True):
    # Use mixed precision matrix multiplication
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Get the resolution the model was trained with
    model, architecture, crop_resolution, norm = model_from_config(args.checkpoint_path)
    args.model = model
    args.architecture = architecture
    args.crop_resolution = crop_resolution
    args.normalization = norm
    model = get_architecture(**args.__dict__).cuda()

    train_loader, test_loader = get_loaders(
        args.dataset,
        args.data_resolution,
        args.n_few_shot,
        args.num_classes,
        args.__dict__,
    )

    print("Loading checkpoint", args.checkpoint_path)
    params = {
        k: v
        for k, v in torch.load(args.checkpoint_path).items()
        if "linear_out" not in k
    }

    print("Load_state output", model.load_state_dict(params, strict=False))

    accuracies = do_few_shot(model, train_loader, test_loader, args.regularization)

    if verbose:
        print(
            "Test accuracies (top1, top5)",
            accuracies,
        )

    del (
        model,
        train_loader,
        test_loader,
    )
    gc.collect()

    return accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scaling MLPs -- few shot")
    # Data
    parser.add_argument(
        "--data_path", default="./beton", type=str, help="Path to data directory"
    )
    parser.add_argument("--dataset", default="cifar100", type=str, help="Dataset")
    parser.add_argument(
        "--data_resolution", default=64, type=int, help="Image Resolution"
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Checkpoint", required=True
    )
    parser.add_argument("--n_few_shot", type=int, default=10)

    ## Optimization
    parser.add_argument("--regularization", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=256, help="Irrelevant here")
    args = parser.parse_args()

    args.num_classes = CLASS_DICT[args.dataset]

    few_shot(args)
