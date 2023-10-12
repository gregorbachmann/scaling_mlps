import torch
from torch import nn


def get_loss_function(args):
    if args.training_objective == "supervised":
        return CrossEntropyLoss(args)
    elif args.training_objective == "reconstruct":
        return ReconstructLoss(args)


class CrossEntropyLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smooth)
        self.args = args

    def forward(self, ims, labels, logits):
        return self.loss_fn(logits, labels)


class ReconstructLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_fn_supervised = nn.CrossEntropyLoss(label_smoothing=args.smooth)
        self.loss_fn_reconstruct = nn.MSELoss()

    def forward(self, ims, labels, preds):
        logits, reconstructs = preds
        supervised_loss = self.loss_fn_supervised(logits, labels)
        reconstructs = reconstructs.view(ims.shape)
        return self.loss_fn_reconstruct(reconstructs, ims) + supervised_loss
