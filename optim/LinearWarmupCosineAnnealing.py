from torch.optim.lr_scheduler import LambdaLR
import torch
import math


def LinearWarmupCosineAnnealing(optimizer, warmup_epochs, T_max, lr_max, lr_min):
    def get_lr(cur_epoch):
        if cur_epoch < warmup_epochs:
            return (cur_epoch + 1) / warmup_epochs
        else:
            return (lr_min + 0.5 * (lr_max - lr_min) * \
                   (1 + math.cos((cur_epoch + 1 - warmup_epochs) * math.pi / (T_max - warmup_epochs)))) / lr_max

    return LambdaLR(optimizer=optimizer, lr_lambda=get_lr)