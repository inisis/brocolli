import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from math import sqrt as sqrt
from itertools import product as product

torch.manual_seed(0)
coco = {
    "num_classes": 201,
    "lr_steps": (280000, 360000, 400000),
    "max_iter": 400000,
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "min_dim": 300,
    "steps": [8, 16, 32, 64, 100, 300],
    "min_sizes": [21, 45, 99, 153, 207, 261],
    "max_sizes": [45, 99, 153, 207, 261, 315],
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    "variance": [0.1, 0.2],
    "clip": True,
    "name": "COCO",
}


class Detect(torch.autograd.Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError("nms_threshold must be non negative.")
        self.conf_thresh = conf_thresh
        self.variance = coco["variance"]

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.nelement() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat(
                    (scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1
                )
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg["min_dim"]
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg["aspect_ratios"])
        self.variance = cfg["variance"] or [0.1]
        self.feature_maps = cfg["feature_maps"]
        self.min_sizes = cfg["min_sizes"]
        self.max_sizes = cfg["max_sizes"]
        self.steps = cfg["steps"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.clip = cfg["clip"]
        self.version = cfg["name"]
        for v in self.variance:
            if v <= 0:
                raise ValueError("Variances must be greater than 0")

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self._is_leaf_module = True
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(1, self.n_channels, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight * x
        return out


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = coco
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.vgg_front = nn.Sequential(*list(base[:23]))
        self.vgg_back = nn.Sequential(*list(base[23:]))
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extra1 = nn.Sequential(*list(extras[0:2]))
        self.extra2 = nn.Sequential(*list(extras[2:4]))
        self.extra3 = nn.Sequential(*list(extras[4:6]))
        self.extra4 = nn.Sequential(*list(extras[6:8]))
        self.extra5 = nn.Sequential(*list(extras[8:10]))
        self.extra6 = nn.Sequential(*list(extras[10:12]))
        self.extra7 = nn.Sequential(*list(extras[12:14]))
        self.extra8 = nn.Sequential(*list(extras[14:16]))

        self.conf_conv1 = nn.Conv2d(
            512, 84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conf_conv2 = nn.Conv2d(
            1024, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conf_conv3 = nn.Conv2d(
            512, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conf_conv4 = nn.Conv2d(
            256, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conf_conv5 = nn.Conv2d(
            256, 84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conf_conv6 = nn.Conv2d(
            256, 84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        self.loc_conv1 = nn.Conv2d(
            512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.loc_conv2 = nn.Conv2d(
            1024, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.loc_conv3 = nn.Conv2d(
            512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.loc_conv4 = nn.Conv2d(
            256, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.loc_conv5 = nn.Conv2d(
            256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.loc_conv6 = nn.Conv2d(
            256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        if phase == "test":
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        x = self.vgg_front(x)

        s = self.L2Norm(x)

        sources.append(s)

        x = self.vgg_back(x)

        sources.append(x)

        x = self.extra1(x)
        x = self.extra2(x)
        sources.append(x)

        x = self.extra3(x)
        x = self.extra4(x)
        sources.append(x)

        x = self.extra5(x)
        x = self.extra6(x)
        sources.append(x)

        x = self.extra7(x)
        x = self.extra8(x)
        sources.append(x)

        loc.append(self.loc_conv1(sources[0]).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_conv1(sources[0]).permute(0, 2, 3, 1).contiguous())

        loc.append(self.loc_conv2(sources[1]).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_conv2(sources[1]).permute(0, 2, 3, 1).contiguous())

        loc.append(self.loc_conv3(sources[2]).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_conv3(sources[2]).permute(0, 2, 3, 1).contiguous())

        loc.append(self.loc_conv4(sources[3]).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_conv4(sources[3]).permute(0, 2, 3, 1).contiguous())

        loc.append(self.loc_conv5(sources[4]).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_conv5(sources[4]).permute(0, 2, 3, 1).contiguous())

        loc.append(self.loc_conv6(sources[5]).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_conv6(sources[5]).permute(0, 2, 3, 1).contiguous())

        # (N, A*K, H, W) to (N, H, W, A*K) to (N, HWAK)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # alternative for the view function
        # loc = torch.cat([torch.flatten(o, start_dim=1) for o in loc], 1)
        # conf = torch.cat([torch.flatten(o, start_dim=1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(
                    conf.view(conf.size(0), -1, self.num_classes)
                ),  # conf preds
                self.priors.type(x.data.type()),  # default boxes
            )
            return output
        else:
            return loc, conf

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == ".pkl" or ".pth":
            print("Loading weights into state dict...")
            self.load_state_dict(
                torch.load(base_file, map_location=lambda storage, loc: storage)
            )
            print("Load weights finished!")
        else:
            print("Sorry only .pth and .pkl files supported.")


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "C":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != "S":
            if v == "S":
                layers += [
                    nn.Conv2d(
                        in_channels,
                        cfg[k + 1],
                        kernel_size=(1, 3)[flag],
                        stride=2,
                        padding=1,
                    )
                ]
                layers += [nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                layers += [nn.ReLU(inplace=True)]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [
            nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)
        ]
        conf_layers += [
            nn.Conv2d(
                vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1
            )
        ]
    for k, v in enumerate(extra_layers[2::4], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [
            nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)
        ]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    "300": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "C",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ],
    "512": [],
}
extras = {
    "300": [256, "S", 512, 128, "S", 256, 128, 256, 128, 256],
    "512": [],
}
mbox = {
    "300": [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    "512": [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train" and phase != "export":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print(
            "ERROR: You specified size "
            + repr(size)
            + ". However, "
            + "currently only SSD300 (size=300) is supported!"
        )
        return
    base_, extras_, head_ = multibox(
        vgg(base[str(size)], 3),
        add_extras(extras[str(size)], 1024),
        mbox[str(size)],
        num_classes,
    )
    return SSD(phase, size, base_, extras_, head_, num_classes)
