import argparse
import csv
import os

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.models as pmodels
import torchvision.transforms as transforms
from PIL import ImageFile

from utils.utils import *
from utils.validation import test_c


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument(
        "--output_prefix",
        default="test",
        type=str,
        help="prefix used to define output path",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="./configs/base_configs.yml",
        type=str,
        metavar="Path",
        help="path to the config file (default: configs.yml)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument("--statedict", default=" ", type=str, help="pre-trained model")
    parser.add_argument(
        "--cleanmodel", action="store_true", help="use model not adversarially trained"
    )
    parser.add_argument("--cleantest", action="store_true", default=None, help="use clean BatchNorms")
    parser.add_argument("--cut", default=1, type=int, help="architexture cut")
    parser.add_argument(
        "--set",
        default="",
        type=str,
        help="select which set to test on (default: all, I(magenet)/ S(tylized)/ C(orrupted)/ (inst)T(a)/ (a)D(vbn))",
    )
    parser.add_argument("--pathI", default=None, type=str, help="path to imagenet")
    parser.add_argument(
        "--pathS", default=None, type=str, help="path to stylized imagenet"
    )
    parser.add_argument(
        "--pathK", default=None, type=str, help="path to imagenet-sketch"
    )
    parser.add_argument("--pathC", default=None, type=str, help="path to imagenet-C")
    parser.add_argument(
        "--pathT", default=None, type=str, help="path to imagenet-instagram"
    )
    parser.add_argument(
        "--pathD", default=None, type=str, help="path to imagenet-AdvBN"
    )
    return parser.parse_args()


# Parase config file and initiate logging
configs = parse_config_file(parse_args())
logger = initiate_logger(configs.output_name)
cudnn.benchmark = True
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    if configs.set == "":
        configs.set = "ISKCTD"

    logger.info("=> using pre-trained model '{}'".format(configs.TRAIN.arch))

    if configs.cleanmodel:
        modelclass = getattr(pmodels, configs.TRAIN.arch)
        model = modelclass(pretrained=True).cuda()
        print("Use pretrained model from torchvison model_zoo")
        model = torch.nn.DataParallel(model).cuda()

    else:
        if "densenet" in configs.TRAIN.arch:
            import models.densenet as models
        elif "resnet" in configs.TRAIN.arch:
            import models.resnet as models
        create_model = getattr(models, configs.TRAIN.arch)
        model = create_model(pretrained=False, num_classes=1000, cut=configs.TRAIN.cut)
        state_dict = torch.load(configs.statedict)
        state_dict = state_dict["state_dict"]
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(state_dict)

    model.eval()

    test_log = configs.statedict[:-8] + str(configs.set) + ".csv"
    result = open(test_log, "wt", newline="")
    cw = csv.writer(result)
    cw.writerow([test_log])
    cw.writerow(["test", "error top1", "error top5", "normalized error"])

    if "I" in configs.set:
        ####################### clean imagenet #######################
        if configs.cleantest is not None:
            cleantest = configs.cleantest
        else:
            cleantest = True
        testdir = os.path.join(configs.pathI, "val")
        test_iter = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                testdir,
                transforms.Compose(
                    [
                        transforms.Resize(configs.DATA.img_size),
                        transforms.CenterCrop(configs.DATA.crop_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ),
            batch_size=configs.DATA.batch_size,
            shuffle=False,
            num_workers=configs.DATA.workers,
            pin_memory=True,
        )
        result, top5 = test_c(test_iter, model, cleantest)
        print(" Clean Prec@1 {result.avg:.3f}\n".format(result=result))
        cw.writerow(["\t\t\t cleantest \t\t\t"])
        cw.writerow(
            [
                "cleantest",
                str(100 - result.avg.item()),
                str(100 - top5.avg.item()),
                "--",
            ]
        )

    if "S" in configs.set:
        ####################### stylized imagenet #######################
        if configs.cleantest is not None:
            cleantest = configs.cleantest
        else:
            cleantest = False
        testdir = os.path.join(configs.pathS, "val")
        test_iter = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                testdir,
                transforms.Compose(
                    [
                        transforms.CenterCrop(configs.DATA.crop_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ),
            batch_size=configs.DATA.batch_size,
            shuffle=False,
            num_workers=configs.DATA.workers,
            pin_memory=True,
        )
        result, top5 = test_c(test_iter, model, cleantest)
        print(" stylized-imagenet: Prec@1 {result.avg:.3f}\n".format(result=result))
        cw.writerow(["\t\t\t stylized imagenet \t\t\t"])
        cw.writerow(
            [
                "stylized imagenet",
                str(100 - result.avg.item()),
                str(100 - top5.avg.item()),
                "--",
            ]
        )

    if "K" in configs.set:
        ####################### stylized imagenet #######################
        if configs.cleantest is not None:
            cleantest = configs.cleantest
        else:
            cleantest = False
        print("======> preparing data")
        test_iter = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                configs.pathK,
                transforms.Compose(
                    [
                        transforms.Resize(configs.DATA.img_size),
                        transforms.CenterCrop(configs.DATA.crop_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ),
            batch_size=configs.DATA.batch_size,
            shuffle=False,
            num_workers=configs.DATA.workers,
            pin_memory=True,
        )
        print("======> finished loading data")
        result, top5 = test_c(test_iter, model, cleantest)
        print(" Sketch-imagenet: Prec@1 {result.avg:.3f}\n".format(result=result))
        cw.writerow(["\t\t\t Sketch imagenet \t\t\t"])
        cw.writerow(
            [
                "Sketch imagenet",
                str(100 - result.avg.item()),
                str(100 - top5.avg.item()),
                "--",
            ]
        )

    if "C" in configs.set:
        ####################### imagenet C #######################
        if configs.cleantest is not None:
            cleantest = configs.cleantest
        else:
            cleantest = True
        severity = ["1", "2", "3", "4", "5"]
        types = [
            "digital/contrast",
            "digital/elastic_transform",
            "digital/jpeg_compression",
            "digital/pixelate",
            "blur/defocus_blur",
            "blur/glass_blur",
            "blur/motion_blur",
            "blur/zoom_blur",
            "noise/gaussian_noise",
            "noise/impulse_noise",
            "noise/shot_noise",
            "weather/brightness",
            "weather/fog",
            "weather/frost",
            "weather/snow",
        ]
        normalizer = [
            0.853204,
            0.646056,
            0.606500,
            0.717840,
            0.819880,
            0.826268,
            0.785948,
            0.798360,
            0.886428,
            0.922640,
            0.894468,
            0.564592,
            0.819324,
            0.826572,
            0.866816,
        ]
        total_err = 0
        for i, corruption in enumerate(types):
            typedir = os.path.join(configs.pathC, corruption)
            type_err = 0
            for degree in severity:
                testdir = os.path.join(typedir, degree)
                test_iter = torch.utils.data.DataLoader(
                    datasets.ImageFolder(
                        testdir,
                        transforms.Compose(
                            [
                                transforms.CenterCrop(configs.DATA.crop_size),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                                ),
                            ]
                        ),
                    ),
                    batch_size=configs.DATA.batch_size,
                    shuffle=False,
                    num_workers=configs.DATA.workers,
                    pin_memory=True,
                )
                result, top5 = test_c(test_iter, model, cleantest)
                # print(' {:s}/{:s}: Final Error@1 {.3f}'.format(corruption, degree, result=result))
                test_type = corruption + "/" + degree
                cw.writerow(
                    [
                        test_type,
                        str(100 - result.avg.item()),
                        str(100 - top5.avg.item()),
                        "--",
                    ]
                )
                type_err += 100.0 - result.avg.item()
                print(
                    " {:s}/{:s}: Final Error@1 {:.3f}".format(
                        corruption, degree, type_err
                    )
                )
            type_err /= 5 * normalizer[i]
            print("{:s}: normalized error: {:.3f}".format(corruption, type_err))
            cw.writerow([corruption, "--", "--", str(type_err)])
            total_err += type_err
        total_err = total_err / (len(types))
        print(
            "mean normalised Corruption Error over 15 types: {:.3f}".format(total_err)
        )
        cw.writerow(["\t\t\t imagenet-C \t\t\t"])
        cw.writerow(["mCE", "--", "--", str(total_err)])

    if "T" in configs.set:
        ####################### insta imagenet #######################
        if configs.cleantest is not None:
            cleantest = configs.cleantest
        else:
            cleantest = True
        testdir = configs.pathT
        filters = [f.path for f in os.scandir(testdir) if f.is_dir()]
        total_err_1 = 0
        total_err_5 = 0
        for filter in filters:
            subdir = os.path.join(testdir, filter)
            test_iter = torch.utils.data.DataLoader(
                datasets.ImageFolder(
                    subdir,
                    transforms.Compose(
                        [
                            transforms.Resize(configs.DATA.img_size),
                            transforms.CenterCrop(configs.DATA.crop_size),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                            ),
                        ]
                    ),
                ),
                batch_size=configs.DATA.batch_size,
                shuffle=False,
                num_workers=configs.DATA.workers,
                pin_memory=True,
            )
            result, top5 = test_c(test_iter, model, cleantest)
            cw.writerow(
                [filter, str(100 - result.avg.item()), str(100 - top5.avg.item()), "--"]
            )
            total_err_1 += 100.0 - result.avg.item()
            total_err_5 += 100.0 - top5.avg.item()
        total_err_1 = total_err_1 / (len(filters))
        total_err_5 = total_err_5 / (len(filters))
        print(
            "mean Error over 20 filter types: @1 {:.3f}/ @5 {:.3f}".format(
                total_err_1, total_err_5
            )
        )
        cw.writerow(["\t\t\t insta-imagenet \t\t\t"])
        cw.writerow(["mCE", "--", "--", str(total_err_1), "/", str(total_err_5)])

    if "D" in configs.set:
        #######################  Adversarial Domain #######################
        if configs.cleantest is not None:
            cleantest = configs.cleantest
        else:
            cleantest = False
        testdir = configs.pathD
        test_iter = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                testdir,
                transforms.Compose(
                    [
                        # transforms.Resize(configs.DATA.img_size),
                        transforms.CenterCrop(configs.DATA.crop_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ),
            batch_size=configs.DATA.batch_size,
            shuffle=False,
            num_workers=configs.DATA.workers,
            pin_memory=True,
        )
        result, top5 = test_c(test_iter, model, cleantest)
        print(" adversarial domain: Prec@1 {result.avg:.3f}\n".format(result=result))
        cw.writerow(["\t\t\t  adversarial domain \t\t\t"])
        cw.writerow(
            [
                "Adversarial Domain imagenet",
                str(100 - result.avg.item()),
                str(100 - top5.avg.item()),
                "--",
            ]
        )


if __name__ == "__main__":
    main()
