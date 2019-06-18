import re
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from model.utils import get_norm


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)  # noqa


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_type='Unknown'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = get_norm(norm_type, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = get_norm(norm_type, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_type='Unknown'):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.norm1 = get_norm(norm_type, planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.norm2 = get_norm(norm_type, planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.norm3 = get_norm(norm_type, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetFPN(nn.Module):

    def __init__(self, block, layers, num_classes=1000, norm_type='Unknown', zero_init_residual=False):  # noqa
        super(ResNetFPN, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.norm1 = get_norm(norm_type, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       norm_type=norm_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       norm_type=norm_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       norm_type=norm_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       norm_type=norm_type)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=2,
                                       norm_type=norm_type)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=False)

        self.merge1 = conv3x3(512, 512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # noqa
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.  # noqa
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677  # noqa
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.norm3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.norm2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1,
                    norm_type='Unknown'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                get_norm(norm_type, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            norm_type))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_type=norm_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        down64 = self.layer5(x)
        up32 = self.up2(down64)
        merge32 = self.merge1(up32 + x)
        out = F.relu(merge32, inplace=True)
        return out


def resnet18fpn5(cfg, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNetFPN(BasicBlock, [2, 2, 2, 2, 2], norm_type=cfg.norm_type,
                      **kwargs)
    if cfg.pretrained:
        pattern = re.compile(r'^(.*bn\d\.(?:weight|bias|running_mean|running_var))$')  # noqa
        state_dict = model_zoo.load_url(model_urls['resnet34'])

        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1).replace('bn', 'norm')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        model.load_state_dict(state_dict, strict=False)
    return model


def resnet34fpn5(cfg, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNetFPN(BasicBlock, [3, 4, 6, 3, 2], norm_type=cfg.norm_type,
                      **kwargs)
    if cfg.pretrained:
        pattern = re.compile(r'^(.*bn\d\.(?:weight|bias|running_mean|running_var))$')  # noqa
        state_dict = model_zoo.load_url(model_urls['resnet34'])

        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1).replace('bn', 'norm')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        model.load_state_dict(state_dict, strict=False)
    return model


def resnet50fpn5(cfg, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFPN(Bottleneck, [3, 4, 6, 3, 2], norm_type=cfg.norm_type,
                      **kwargs)
    if cfg.pretrained:
        pattern = re.compile(r'^(.*bn\d\.(?:weight|bias|running_mean|running_var))$')  # noqa
        state_dict = model_zoo.load_url(model_urls['resnet34'])

        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1).replace('bn', 'norm')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        model.load_state_dict(state_dict, strict=False)
    return model


def resnet101fpn5(cfg, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFPN(Bottleneck, [3, 4, 23, 3, 2], norm_type=cfg.norm_type,
                      **kwargs)
    if cfg.pretrained:
        pattern = re.compile(r'^(.*bn\d\.(?:weight|bias|running_mean|running_var))$')  # noqa
        state_dict = model_zoo.load_url(model_urls['resnet34'])

        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1).replace('bn', 'norm')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        model.load_state_dict(state_dict, strict=False)
    return model


def resnet152fpn5(cfg, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFPN(Bottleneck, [3, 8, 36, 3, 2], norm_type=cfg.norm_type,
                      **kwargs)
    if cfg.pretrained:
        pattern = re.compile(r'^(.*bn\d\.(?:weight|bias|running_mean|running_var))$')  # noqa
        state_dict = model_zoo.load_url(model_urls['resnet34'])

        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1).replace('bn', 'norm')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        model.load_state_dict(state_dict, strict=False)
    return model
