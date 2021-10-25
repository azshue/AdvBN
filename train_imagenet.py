# This module is adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
import os
import random
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.data_parallel import DataParallel

from utils.utils import *
from utils.validation import validate


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "--output_prefix",
        default="std_adv",
        type=str,
        help="prefix used to define output path",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs.yml",
        type=str,
        metavar="Path",
        help="path to the config file (default: configs.yml)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--load",
        default="",
        type=str,
        metavar="PATH",
        help="path to pretrained weight (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--nrepeat",
        default=None,
        type=int,
        help="overwrite the #adversarial-repeat in configs.yml",
    )
    parser.add_argument(
        "--lr_step",
        default=10,
        type=int,
        help="after how many iters, decrease lr by lr_factor",
    )
    parser.add_argument("--lr_factor", default=0.1, type=float)
    parser.add_argument("--adv_step", default=None, type=float, help="fgsm step size")
    parser.add_argument(
        "--eps", default=None, type=float, help="adversarial step: projection radias"
    )
    parser.add_argument("--cut", default=1, type=int)
    return parser.parse_args()


best_prec1 = 0


def main():
    # Parase config file and initiate logging
    configs = parse_config_file(parse_args())
    logger = initiate_logger(configs.output_name)
    # print = logger.info

    # Create output folder
    if not os.path.isdir(os.path.join("trained_models", configs.output_name)):
        os.makedirs(os.path.join("trained_models", configs.output_name))
    # Log the config details
    logger.info(pad_str(" ARGUMENTS "))
    for k, v in configs.items():
        logger.info("{}: {}".format(k, v))
    logger.info(pad_str(""))

    ngpus_per_node = torch.cuda.device_count()
    world_size = 1
    configs.world_size = ngpus_per_node * world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, configs))


def main_worker(gpu, ngpus_per_node, configs):
    # Scale and initialize the parameters
    global best_prec1
    configs.gpu = gpu
    if configs.gpu is not None:
        print("Use GPU: {} for training".format(configs.gpu))
    configs.rank = 0
    configs.ngpus_per_node = ngpus_per_node
    configs.rank = configs.rank * ngpus_per_node + gpu
    dist_url = "tcp://localhost:" + str(
        8000 + (int(time.time() % 1000)) // 10
    )
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=configs.world_size,
        rank=configs.rank,
    )

    # Create the model
    if configs.gpu == 0:
        print("=> using model architecture '{}'".format(configs.TRAIN.arch))
    if "resnet" in configs.TRAIN.arch:
        import models.resnet as resnet
        create_model = getattr(resnet, configs.TRAIN.arch)
    elif "densenet" in configs.TRAIN.arch:
        import models.densenet as densenet
        create_model = getattr(densenet, configs.TRAIN.arch)
    else:
        raise NotImplementedError
    model = create_model(pretrained=True, num_classes=1000, cut=configs.TRAIN.cut)
    for k, v in model.state_dict().items():
        if torch.isnan(v).any():
            print('{} has nan'.format(k))
    # Use weights other than the pytorch ones for initialization
    if configs.load:
        if os.path.isfile(configs.load):
            print("=> loading pretrained weight '{}'".format(configs.load))
            checkpoint = torch.load(configs.load)
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            if not isinstance(model, (DataParallel, DistributedDataParallel)):
                model.load_state_dict(
                    {k.replace("module.", ""): v for k, v in checkpoint.items()}
                )
            else:
                model.load_state_dict(
                    {
                        k if "module." in k else "module." + k: v
                        for k, v in checkpoint.items()
                    }
                )

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap the model into DDP
    if configs.gpu is not None:
        torch.cuda.set_device(configs.gpu)
        model.cuda(configs.gpu)
        configs.DATA.batch_size = int(configs.DATA.batch_size / ngpus_per_node)
        configs.DATA.workers = int((configs.DATA.workers) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[configs.gpu],
            find_unused_parameters=False,
            broadcast_buffers=True,
        )

    # Criterion:
    criterion = nn.CrossEntropyLoss().cuda(configs.gpu)

    # Optimizer:
    if configs.TRAIN.optim == "SGD":
        optimizer = torch.optim.SGD(
            model.module.head.parameters(),
            configs.TRAIN.lr,
            momentum=configs.TRAIN.momentum,
            weight_decay=configs.TRAIN.weight_decay,
        )
    else:
        raise NotImplementedError

    # Resume if a valid checkpoint path is provided
    if configs.resume:
        if os.path.isfile(configs.resume):
            print("=> loading checkpoint '{}'".format(configs.resume))
            if configs.gpu is None:
                checkpoint = torch.load(configs.resume)
            else:
                loc = "cuda:{}".format(configs.gpu)
                checkpoint = torch.load(configs.resume, map_location=loc)
            configs.TRAIN.start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            if configs.gpu is not None:
                best_prec1 = best_prec1.to(configs.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    configs.resume, checkpoint["epoch"]
                )
            )
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(configs.resume))

    cudnn.benchmark = True

    # Initiate data loaders
    traindir = os.path.join(configs.data, "train")
    valdir = os.path.join(configs.data, "val")

    transform_list = [
        transforms.RandomResizedCrop(configs.DATA.crop_size),
        transforms.RandomHorizontalFlip(),
    ]
    preprocess = [
        transforms.ToTensor(),
        transforms.Normalize(configs.TRAIN.mean, configs.TRAIN.std),
    ]

    transform_list.extend(preprocess)
    if configs.gpu == 0:
        print("=> start loading training data ")

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose(transform_list))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs.DATA.batch_size,
        shuffle=(train_sampler is None),
        num_workers=configs.DATA.workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    if configs.gpu == 0:
        print("=> finished loading training data ")

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(configs.DATA.img_size),
                    transforms.CenterCrop(configs.DATA.crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(configs.TRAIN.mean, configs.TRAIN.std),
                ]
            ),
        ),
        batch_size=configs.DATA.batch_size,
        shuffle=False,
        num_workers=configs.DATA.workers,
        pin_memory=True,
    )

    for epoch in range(configs.TRAIN.start_epoch, configs.TRAIN.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(
            configs.TRAIN.lr, optimizer, epoch, configs.lr_step, configs.lr_factor
        )

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, configs)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, configs, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if configs.rank % ngpus_per_node == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": configs.TRAIN.arch,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                os.path.join("trained_models", configs.output_name),
            )


def train(train_loader, model, criterion, optimizer, epoch, configs):
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_a = AverageMeter()
    losses_c = AverageMeter()
    top1_clean = AverageMeter()
    top5_clean = AverageMeter()
    top1_adv = AverageMeter()
    top5_adv = AverageMeter()
    # switch to train mode
    model.train()
    for m in model.module.feature_x.modules():
        if isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d, nn.GroupNorm)):
            m.eval()
    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        if configs.gpu is not None:
            input = input.cuda(configs.gpu, non_blocking=True)
        target = target.cuda(configs.gpu, non_blocking=True)
        data_time.update(time.time() - end)

        for param in model.parameters():
            param.grad = None

        with torch.no_grad():
            feature = model(input, "feature_x")

        torch.autograd.set_detect_anomaly(True)
        clean_feature = feature.detach().clone().requires_grad_(True)
        adv_feature = perturb(model, feature, target, criterion, configs)
        adv_output = model(adv_feature, "head", "adv")
        loss_a = criterion(adv_output, target)
        clean_output = model(clean_feature, "head", "clean")
        loss_c = criterion(clean_output, target)

        loss = loss_c + loss_a
        clean_prec1, clean_prec5 = accuracy(clean_output, target, topk=(1, 5))
        adv_prec1, adv_prec5 = accuracy(adv_output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses_a.update(loss_a.item(), input.size(0))
        losses_c.update(loss_c.item(), input.size(0))
        top1_clean.update(clean_prec1[0], input.size(0))
        top5_clean.update(clean_prec5[0], input.size(0))
        top1_adv.update(adv_prec1[0], input.size(0))
        top5_adv.update(adv_prec5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if (
            configs.rank % configs.ngpus_per_node == 0
            and i % configs.TRAIN.print_freq == 0
        ):
            print(
                "Train Iter: [{0}/{1}][{2}/{3}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t"
                "clean_Prec@1 {top1_c.val:.3f} ({top1_c.avg:.3f})\t"
                "clean_Prec@5 {top5_c.val:.3f} ({top5_c.avg:.3f})\t"
                "adv_Prec@1 {top1_a.val:.3f} ({top1_a.avg:.3f})\t"
                "adv_Prec@5 {top5_a.val:.3f} ({top5_a.avg:.3f})".format(
                    epoch,
                    configs.TRAIN.epochs,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    top1_c=top1_clean,
                    top5_c=top5_clean,
                    top1_a=top1_adv,
                    top5_a=top5_adv,
                    cls_loss=losses,
                )
            )
            sys.stdout.flush()


def perturb(model, feature, target, criterion, configs):
    size = feature.size()
    noise_size = [1, size[1], 1, 1]
    noise_batch_mean = Variable(torch.ones(noise_size)).cuda(configs.gpu)
    noise_batch_std = Variable(torch.ones(noise_size)).cuda(configs.gpu)

    ori_mean, ori_std = calc_mean_std(feature)
    ori_mean = ori_mean.cuda(configs.gpu)
    ori_std = ori_std.cuda(configs.gpu)
    normalized_feature = feature - ori_mean
    normalized_feature.detach_()
    model.eval()

    with torch.enable_grad():
        for _iter in range(configs.ADV.n_repeats):
            noise_batch_mean.requires_grad_(True)
            noise_batch_std.requires_grad_(True)

            new_mean = ori_mean * noise_batch_mean
            new_std = noise_batch_std
            adv_feature = normalized_feature * new_std + new_mean

            if "densenet" in configs.TRAIN.arch:
                input_feature = adv_feature
            else:
                # relu layer in resnet
                input_feature = torch.clamp(adv_feature, min=0.0)

            output = model(input_feature, "head", "adv")
            loss = criterion(output, target)
            grads_mean, grads_std = torch.autograd.grad(
                loss,
                [noise_batch_mean, noise_batch_std],
                grad_outputs=None,
                only_inputs=True,
                allow_unused=True,
            )[:2]

            # adversarial step
            noise_batch_mean.data += configs.ADV.fgsm_step * torch.sign(grads_mean.data)
            noise_batch_std.data += configs.ADV.fgsm_step * torch.sign(grads_std.data)

            # projection
            noise_batch_mean = torch.clamp(
                noise_batch_mean,
                min=1 - configs.ADV.scale_eps,
                max=1 + configs.ADV.scale_eps,
            )
            noise_batch_std = torch.clamp(
                noise_batch_std,
                min=1 - configs.ADV.scale_eps,
                max=1 + configs.ADV.scale_eps,
            )

            adv_feature.detach_()
            noise_batch_mean.detach_()
            noise_batch_std.detach_()

    new_mean = ori_mean * noise_batch_mean
    new_std = noise_batch_std

    adv_feature = normalized_feature * new_std + new_mean
    if "densenet" in configs.TRAIN.arch:
        out_feature = adv_feature
    else:
        out_feature = torch.clamp(adv_feature, min=0.0)

    model.train()
    for m in model.module.feature_x.modules():
        if isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d, nn.GroupNorm)):
            m.eval()
    return out_feature.detach()


if __name__ == "__main__":
    main()
