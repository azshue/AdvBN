from collections import OrderedDict

import torch
from torch.nn.modules.batchnorm import BatchNorm2d


def convert_ckpt(model, weight, adv_key="adv_bn"):
    """
    Args:
        model (torch.nn.Module): a model with additional adv_bn layers
        weight (state_dict): the statedict without adv_bn layers
        adv_key (str): the keyword for adv_bn layers in the model
    """
    sd = model.state_dict()
    kv = weight

    values = []
    kv_keys = []
    for k, value in kv.items():
        kv_keys.append(k)
        values.append(value)

    nkv = OrderedDict()
    index = 0
    for k in sd:
        if adv_key in k:
            # initialize the adv_bn layer using statistcs from its clean counterparts
            nkv[k] = nkv[k.replace(adv_key, adv_key.replace("adv", "clean"))]
        elif adv_key.replace("adv", "clean") in k or 'feature_x.1' in k or \
            isinstance(getattr(model, '.'.join(k.split('.')[:-1])), BatchNorm2d):  
            # in case bn stats are stored in different order (wt/bias, mean/var vs. mean/var, wt/bias) 
            if ('weight' in k and 'weight' not in kv_keys[index]) or \
                ('bias' in k and 'bias' not in kv_keys[index]):
                nkv[k] = values[index+2]
                index += 1
            elif ('mean' in k and 'mean' not in kv_keys[index]) or \
                ('var' in k and 'var' not in kv_keys[index]):
                nkv[k] = values[index-2]
                index += 1
            elif "num_batches_tracked" in k and "num_batched_tracked" not in kv_keys[index]:
                # for converting models weights from older pytorch version
                nkv[k] = torch.BoolTensor([True])
            else:
                nkv[k] = values[index]
                index += 1
        elif "num_batches_tracked" in k and "num_batched_tracked" not in kv_keys[index]:
            # for converting models weights from older pytorch version
            nkv[k] = torch.BoolTensor([True])
        else:
            nkv[k] = values[index]
            index += 1

    return nkv
