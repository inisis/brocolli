from torch import nn

import torch.nn.functional as F
from model.backbone.vgg import (vgg19, vgg19_bn)
from model.backbone.resnet import (resnet18, resnet34, resnet50, resnet101,
                                   resnet152)
from model.backbone.resnetfpn5 import (resnet18fpn5, resnet34fpn5,
                                       resnet50fpn5, resnet101fpn5,
                                       resnet152fpn5)
from model.backbone.resnetfpn6 import (resnet18fpn6, resnet34fpn6,
                                       resnet50fpn6, resnet101fpn6,
                                       resnet152fpn6)
from model.backbone.densenet import (densenet121, densenet169, densenet201)
from model.backbone.inception import (inception_v3)
from model.global_pool import GlobalPool
from model.attention_map import AttentionMap


BACKBONES = {'vgg19': vgg19,
             'vgg19_bn': vgg19_bn,
             'resnet18': resnet18,
             'resnet34': resnet34,
             'resnet50': resnet50,
             'resnet101': resnet101,
             'resnet152': resnet152,
             'resnet18fpn5': resnet18fpn5,
             'resnet34fpn5': resnet34fpn5,
             'resnet50fpn5': resnet50fpn5,
             'resnet101fpn5': resnet101fpn5,
             'resnet152fpn5': resnet152fpn5,
             'resnet18fpn6': resnet18fpn6,
             'resnet34fpn6': resnet34fpn6,
             'resnet50fpn6': resnet50fpn6,
             'resnet101fpn6': resnet101fpn6,
             'resnet152fpn6': resnet152fpn6,
             'densenet121': densenet121,
             'densenet169': densenet169,
             'densenet201': densenet201,
             'inception_v3': inception_v3}


BACKBONES_TYPES = {'vgg19': 'vgg',
                   'vgg19_bn': 'vgg',
                   'resnet18': 'resnet',
                   'resnet34': 'resnet',
                   'resnet50': 'resnet',
                   'resnet101': 'resnet',
                   'resnet152': 'resnet',
                   'resnet18fpn5': 'resnet',
                   'resnet34fpn5': 'resnet',
                   'resnet50fpn5': 'resnet',
                   'resnet101fpn5': 'resnet',
                   'resnet152fpn5': 'resnet',
                   'resnet18fpn6': 'resnet',
                   'resnet34fpn6': 'resnet',
                   'resnet50fpn6': 'resnet',
                   'resnet101fpn6': 'resnet',
                   'resnet152fpn6': 'resnet',
                   'densenet121': 'densenet',
                   'densenet169': 'densenet',
                   'densenet201': 'densenet',
                   'inception_v3': 'inception'}


class Classifier(nn.Module):

    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.cfg = cfg
        self.backbone = BACKBONES[cfg.backbone](cfg)
        self.global_pool = GlobalPool(cfg)
        self._init_classifier()
        self._init_attention_map()

    def _init_classifier(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                setattr(self, "fc_" + str(index),
                        nn.Linear(512, num_class))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'resnet':
                setattr(self, "fc_" + str(index),
                        nn.Linear(512 * self.backbone.block.expansion,
                                  num_class))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                setattr(self, "fc_" + str(index),
                        nn.Linear(self.backbone.num_features, num_class))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                setattr(self, "fc_" + str(index),
                        nn.Linear(2048, num_class))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.cfg.backbone)
                )

            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Linear):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def _init_attention_map(self):
        if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
            setattr(self, "attention_map", AttentionMap(self.cfg, 512))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'resnet':
            setattr(self, "attention_map", AttentionMap(self.cfg,
                    512 * self.backbone.block.expansion))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
            setattr(self, "attention_map", AttentionMap(self.cfg,
                    self.backbone.num_features))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
            setattr(self, "attention_map", AttentionMap(self.cfg, 2048))
        else:
            raise Exception(
                'Unknown backbone type : {}'.format(self.cfg.backbone)
            )

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, x):
        feat_map = self.backbone(x)
        if self.cfg.attention_map != "None":
            feat_map = self.attention_map(feat_map)
        outs = list()
        for index, num_class in enumerate(self.cfg.num_classes):
            classifier = getattr(self, "fc_" + str(index))
            feat_map_ = feat_map.permute(0, 2, 3, 1)
            # logit_map = (N, C, H, W)
            logit_map = classifier(feat_map_).permute(0, 3, 1, 2)

            feat = self.global_pool(feat_map, logit_map)
            feat = F.dropout(feat, p=self.cfg.fc_drop, training=self.training)
            feat = feat.view(feat.size(0), -1)
            out = classifier(feat)
            outs.append(out)
        return outs
