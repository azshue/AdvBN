import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

from torch import Tensor
from torch.jit.annotations import List

from utils.convert_weight import convert_ckpt

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False, first_layer=False):
        super(_DenseLayer, self).__init__()
        if not first_layer:
            self.add_module('clean_norm1', nn.BatchNorm2d(num_input_features)),
            self.add_module('adv_norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('clean_norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('adv_norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient
        self.first_layer = first_layer

    def bn_function(self, inputs, tag='clean'):
        concated_features = torch.cat(inputs, 1)
        if self.first_layer:
            input_feature = concated_features
        else:
            if tag == 'clean':
                input_feature = self.clean_norm1(concated_features)
            else:
                input_feature = self.adv_norm1(concated_features)
        bottleneck_output = self.conv1(self.relu1(input_feature))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input, tag='clean'):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features, tag)

        if tag == 'clean':
            input_feature = self.clean_norm2(bottleneck_output)
        else:
            input_feature = self.adv_norm2(bottleneck_output)

        new_features = self.conv2(self.relu2(input_feature))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False, first_block=False):
        super(_DenseBlock, self).__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                first_layer=(first_block and i==0)
            )
            setattr(self, 'denselayer%d' % (i + 1), layer)

    def forward(self, init_features, tag='clean'):
        features = [init_features]
        for i in range(self.num_layers):
            m = getattr(self, 'denselayer%d' % (i + 1))
            new_features = m(features, tag)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.clean_norm = nn.BatchNorm2d(num_input_features)
        self.adv_norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x, tag='clean'):
        if tag == 'clean':
            x = self.clean_norm(x)
        else:
            x = self.adv_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class Head(nn.Module):
    def __init__(self, num_features, growth_rate, block_config, 
                bn_size, drop_rate, num_classes, memory_efficient):
        super(Head, self).__init__()
        num_features = num_features
        self.block_config = block_config
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                first_block=(i==0)
            )
            setattr(self, 'denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                setattr(self, 'transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.clean_norm5 = nn.BatchNorm2d(num_features)
        self.adv_norm5 = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x, tag='clean'):
        for i in range(len(self.block_config)):
            m1 = getattr(self, 'denseblock%d' % (i + 1))
            x = m1(x, tag)
            if i != len(self.block_config) - 1:
                m2 = getattr(self, 'transition%d' % (i + 1))
                x = m2(x, tag)
        if tag == 'clean':
            x = self.clean_norm5(x)
        else:
            x = self.adv_norm5(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class FeatureX(nn.Module):
    def __init__(self, num_features, growth_rate, block_config, 
                bn_size, drop_rate, num_classes, memory_efficient):
        super(FeatureX, self).__init__()
        num_init_features = num_features
        self.conv0 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(num_init_features)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block_config = block_config
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            setattr(self, 'denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            trans = _Transition(num_input_features=num_features,
                                num_output_features=num_features // 2)
            setattr(self, 'transition%d' % (i + 1), trans)
            num_features = num_features // 2
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        for i in range(len(self.block_config)):
            m1 = getattr(self, 'denseblock%d' % (i + 1))
            x = m1(x)
            m2 = getattr(self, 'transition%d' % (i + 1))
            x = m2(x)
        return x

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),num_init_features=64, 
                bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False, 
                cut=1, train_all=False):
        super(DenseNet, self).__init__()
        # Each denseblock
        num_features = num_init_features
        self.feature_x = FeatureX(num_features, growth_rate, block_config[:int(cut)], 
                        bn_size, drop_rate, num_classes, memory_efficient)
        # fix the g^{1, l} when training with advbn
        if not train_all:
            for params in getattr(self, 'feature_x').parameters():
                params.requires_grad = False
            for m in self.feature_x.modules():
                   if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                       m.eval()

        for i, num_layers in enumerate(block_config[:int(cut)]):
            num_features = num_features + num_layers * growth_rate
            num_features = num_features // 2

        self.head = Head(num_features, growth_rate, block_config[int(cut):], 
                        bn_size, drop_rate, num_classes, memory_efficient)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x, stage, tag='clean'):
        if stage == 'feature_x':
            x = self.feature_x(x)
        elif stage == 'head':
            x = self.head(x, tag)
        else:
            x = self.feature_x(x)
            x = self.head(x, tag)
        return x

    def forward(self, x, stage, tag='clean'):
        return self._forward_impl(x, stage, tag)


def _load_state_dict(model, model_url):
    kv = torch.hub.load_state_dict_from_url(model_url)
    nkv = convert_ckpt(model, kv, 'adv_norm')
    model.load_state_dict(nkv)


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch])
    return model


def densenet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)


def densenet161(pretrained=False, progress=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)


def densenet169(pretrained=False, progress=True, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     **kwargs)


def densenet201(pretrained=False, progress=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)
