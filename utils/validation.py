import sys
import time

import numpy as np
import torch

from .utils import *


def validate(val_loader, model, criterion, configs, epoch):
    # Mean/Std for normalization
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()

    # Initiate the meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            if hasattr(configs, "gpu"):
                if configs.gpu is not None:
                    input = input.cuda(configs.gpu, non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input, "full")
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % configs.TRAIN.print_freq == 0:
                if hasattr(configs, "rank") and hasattr(configs, "ngpus_per_node"):
                    if not configs.rank % configs.ngpus_per_node == 0:
                        continue
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )
                sys.stdout.flush()

    print(
        " Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(
            top1=top1, top5=top5
        )
    )
    return top1.avg


def test_c(dataloader, model, cleantest):
    top1 = AverageMeter()
    top5 = AverageMeter()
    for i, (input, target) in enumerate(dataloader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            if cleantest:
                output = model(input, "full", "clean")
            else:
                output = model(input, "full", "adv")

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
    return top1, top5
