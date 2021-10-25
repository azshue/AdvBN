import argparse
import os

import models.autoencoder as net
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from utils.utils import *

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument(
    "--sourcedir", type=str, required=True, help="Directory path to the clean test set"
)
parser.add_argument(
    "--targetdir",
    type=str,
    required=True,
    help="Directory path to store the adv test set",
)
parser.add_argument("--vgg", type=str, help="pretrained vgg encoder")
parser.add_argument("--decoder", type=str, help="pretrained decoder")
parser.add_argument("--batchsize", type=int)
parser.add_argument("--nrepeat", type=int, default=4)
parser.add_argument("--step", type=float, default=0.15)
parser.add_argument("--eps", type=float, default=0.4)
parser.add_argument(
    "--identical", action="store_true", default=False, help="identical mapping"
)

args = parser.parse_args()

vgg = net.vgg
vgg.load_state_dict(torch.load(args.vgg))
encoder = nn.Sequential(*list(vgg.children())[:31])
decoder = net.decoder
decoder.load_state_dict(torch.load(args.decoder))
model = net.Net(encoder, decoder)
model = torch.nn.DataParallel(model).cuda()
predict_net = torchvision.models.vgg19(pretrained=True)

predict_net = torch.nn.DataParallel(predict_net).cuda()
model.eval()
predict_net.eval()

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        args.sourcedir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ]
        ),
    ),
    batch_size=args.batchsize,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
)

counter = 0
current_class = None
current_class_files = None

all_classes = sorted(os.listdir(args.sourcedir))

for i, (input, label) in enumerate(test_loader):
    input = input.cuda(non_blocking=True)
    label = label.cuda(non_blocking=True)
    features = model(input, "encode")
    features.detach_()
    size = features.size()

    if args.identical:
        clean_feature = features.detach().clone().cuda()
        input_features = clean_feature
    else:
        noise_size = [1, size[1], 1, 1]
        noise_batch_mean = Variable(torch.ones(noise_size)).cuda()
        noise_batch_std = Variable(torch.ones(noise_size)).cuda()

        predict_net.eval()
        criterion = nn.CrossEntropyLoss().cuda()

        mean = torch.Tensor(np.array([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis])
        mean = mean.expand(3, 256, 256).cuda()
        std = torch.Tensor(np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis])
        std = std.expand(3, 256, 256).cuda()

        with torch.enable_grad():
            for _iter in range(args.nrepeat):
                features.requires_grad_(True)
                noise_batch_mean.requires_grad_(True)
                noise_batch_std.requires_grad_(True)

                ori_mean, ori_std = calc_mean_std(features)
                new_mean = ori_mean.cuda() * noise_batch_mean
                new_std = noise_batch_std
                normalized_feature = features - ori_mean.expand(size)
                adv_feature = normalized_feature * new_std.expand(
                    size
                ) + new_mean.expand(size)
                input_feature = torch.clamp(adv_feature, min=0.0)
                adv_image = model(input_feature, "decode")

                # transform the image for testing on pretrained pytorch vgg-19 models
                adv_image.sub_(mean).div_(std)
                pred = predict_net(adv_image)
                loss = criterion(pred, label)

                grads_mean, grads_std = torch.autograd.grad(
                    loss,
                    [noise_batch_mean, noise_batch_std],
                    grad_outputs=None,
                    only_inputs=True,
                )[:2]

                noise_batch_mean.data += args.step * torch.sign(grads_mean.data)
                noise_batch_std.data += args.step * torch.sign(grads_std.data)

                # projection
                noise_batch_mean = torch.clamp(
                    noise_batch_mean, min=1 - args.eps, max=1 + args.eps
                )
                noise_batch_std = torch.clamp(
                    noise_batch_std, min=1 - args.eps, max=1 + args.eps
                )

                features.detach_()
                noise_batch_mean.detach_()
                noise_batch_std.detach_()

        new_mean = ori_mean.cuda() * noise_batch_mean
        new_std = noise_batch_std
        normalized_feature = features - ori_mean.expand(size)
        features = normalized_feature * new_std.expand(size) + new_mean.expand(size)
        features = torch.clamp(features, min=0.0)
        input_features = features

    image = model(input_features, "decode")

    for img_index in range(image.size()[0]):
        source_class = all_classes[label[img_index]]
        source_classdir = os.path.join(args.sourcedir, source_class)
        assert os.path.exists(source_classdir)

        target_classdir = os.path.join(args.targetdir, source_class)
        if not os.path.exists(target_classdir):
            os.makedirs(target_classdir)

        if source_class != current_class:
            # moving on to new class:
            # start counter (=index) by 0, update list of files
            # for this new class
            counter = 0
            current_class_files = sorted(os.listdir(source_classdir))

        current_class = source_class

        target_img_path = os.path.join(
            target_classdir, current_class_files[counter].replace(".JPEG", ".png")
        )
        save_image(image[img_index, :, :, :], target_img_path)
        counter += 1

    if i % 100 == 0:
        print("Progress: [{0}/{1}]\t".format(i, len(test_loader)))
