import datetime
import logging
# import torchvision.models as models
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict
from PIL import Image

## decide the #channels of a feature map given an model and the cutting layer
feat_channel_dict = {"resnet50": {1: 256, 2: 512, 3: 1024},
                     "densenet121":{1: 128}}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(initial_lr, optimizer, epoch, lr_step, lr_factor):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (lr_factor ** (epoch // int(lr_step)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def fgsm(gradz, step_size):
    return step_size * torch.sign(gradz)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def initiate_logger(output_path):
    if not os.path.isdir(os.path.join("output", output_path)):
        os.makedirs(os.path.join("output", output_path))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join("output", output_path, "log.txt"), "w")
    )
    logger.info(pad_str(" LOGISTICS "))
    logger.info(
        "Experiment Date: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    logger.info("Output Name: {}".format(output_path))
    logger.info("User: {}".format(os.getenv("USER")))
    return logger


# def get_model_names():
# 	return sorted(name for name in models.__dict__
#     		if name.islower() and not name.startswith("__")
#     		and callable(models.__dict__[name]))


def pad_str(msg, total_len=70):
    rem_len = total_len - len(msg)
    return "*" * int(rem_len / 2) + msg + "*" * int(rem_len / 2)


def parse_config_file(args):
    with open(args.config) as f:
        config = EasyDict(yaml.load(f))

    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v
    if hasattr(config, "nrepeat"):
        if config.nrepeat is not None:
            config.ADV.n_repeats = config.nrepeat
    if hasattr(config, "adv_step"):
        if config.adv_step is not None:
            config.ADV.fgsm_step = config.adv_step
    if hasattr(config, "eps"):
        if config.eps is not None:
            config.ADV.scale_eps = config.eps
    if hasattr(config, "cut"):
        config.TRAIN.cut = config.cut

    # Add the output path
    if "autoencoder" in args.output_prefix:
        config.output_name = "{:s}_lr{:e}tv_{:e}_bs{:d}".format(
            args.output_prefix,
            float(args.lr),
            float(args.tv_weight),
            int(args.batch_size),
        )
    else:
        config.output_name = "{:s}_lr{:.3f}_optim{:s}_wd{:.4f}_mom{:.3f}_repeat{:d}_step{:.3f}_eps{:.3f}".format(
            args.output_prefix,
            float(config.TRAIN.lr),
            str(config.TRAIN.optim),
            float(config.TRAIN.weight_decay),
            float(config.TRAIN.momentum),
            config.ADV.n_repeats,
            float(config.ADV.fgsm_step),
            float(config.ADV.scale_eps),
        )
    config.feat_channels = feat_channel_dict[config.TRAIN.arch][config.TRAIN.cut]

    return config


def save_checkpoint(state, is_best, filepath):
    filename = os.path.join(filepath, "checkpoint.pth.tar")
    # Save model
    torch.save(state, filename)
    if int(state["epoch"]) % 10 == 0:
        filename2 = os.path.join(
            filepath, "checkpoint_{:d}.pth.tar".format(int(state["epoch"]))
        )
        torch.save(state, filename2)
    # Save best model
    if is_best:
        shutil.copyfile(filename, os.path.join(filepath, "model_best.pth.tar"))


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    C = size[1]

    feat_mean = feat.permute(1, 0, 2, 3).reshape(C, -1).mean(dim=1).view(1, C, 1, 1)
    feat_var = feat.permute(1, 0, 2, 3).reshape(C, -1).var(dim=1) + eps
    feat_std = feat_var.sqrt().view(1, C, 1, 1)

    return feat_mean, feat_std
