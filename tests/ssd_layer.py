# -*- coding: utf-8 -*-
import sys
import argparse
import json
import collections
from easydict import EasyDict as edict

sys.path.append('/tool/caffe/python')
sys.path.append('/tool/caffe/python/caffe')

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

parser = argparse.ArgumentParser(description='Add ssd layer')
parser.add_argument('cfg_name', default=None, metavar='CFG_NAME', type=str,
                    help="Name to the config file in json format")
parser.add_argument('prototxt_name', default=None, metavar='PROTOTXT_NAME', type=str,
                    help="Name to original prototxt")
parser.add_argument('save_name', default=None, metavar='SAVE_NAME', type=str,
                    help="Name to the transformed prototxt")
parser.add_argument('--verbose', default=False, type=bool, help="Detail info")


def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
        use_scale=True, min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
        flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
        conf_postfix='', loc_postfix='', **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"

    #assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    
    loc_layers = []
    conf_layers = []

    priorbox_layers = collections.OrderedDict()
    norm_name_layers = collections.OrderedDict()
    for i in range(0, num):
        from_layer = from_layers[i]
        # Get the normalize value.
        if normalizations: 
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                norm_name_layers[norm_name] = net.layer.add()
                norm_name_layers[norm_name].CopyFrom(L.Normalize(scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False).to_proto().layer[0])
                norm_name_layers[norm_name].name = norm_name
                norm_name_layers[norm_name].top[0] = norm_name
                norm_name_layers[norm_name].bottom.append(from_layer)
                from_layer = norm_name
    
        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        priorbox_layers[name] = net.layer.add()
        priorbox_layers[name].CopyFrom(L.PriorBox(min_size=min_size,
                                       clip=clip, variance=prior_variance, offset=offset).to_proto().layer[0])

        priorbox_layers[name].name = name
        priorbox_layers[name].top[0] = name
        priorbox_layers[name].bottom.append(from_layer)
        priorbox_layers[name].bottom.append(data_layer)

        if max_size: 
            priorbox_layers[name].prior_box_param.max_size.extend(max_size)
        if aspect_ratio:
            priorbox_layers[name].prior_box_param.aspect_ratio.extend(aspect_ratio)
        if flip:
            priorbox_layers[name].prior_box_param.flip = flip
        if step:
            priorbox_layers[name].prior_box_param.step = step

    # Concatenate priorbox, loc, and conf layers.
    name = "mbox_priorbox"
    cat_mbox_layer = net.layer.add()
    cat_mbox_layer.CopyFrom(L.Concat(axis=2).to_proto().layer[0])
    cat_mbox_layer.name = name
    cat_mbox_layer.top[0] = name
    for bt in priorbox_layers.keys():
        cat_mbox_layer.bottom.append(bt)

def run(args):
    with open(args.cfg_name) as f:
        cfg = edict(json.load(f))

    net = caffe_pb2.NetParameter()
    with open(args.prototxt_name) as f:
        s = f.read()
        txtf.Merge(s, net)

    CreateMultiBoxHead(net, data_layer='data', from_layers=cfg.mbox_source_layers,
            use_batchnorm=False, min_sizes=cfg.min_sizes, max_sizes=cfg.max_sizes,
            aspect_ratios=cfg.aspect_ratios, steps=cfg.steps, normalizations=cfg.normalizations,
            num_classes=cfg.num_classes, share_location=cfg.share_location, flip=cfg.flip, clip=cfg.clip,
            prior_variance=cfg.prior_variance, kernel_size=3, pad=1, lr_mult=1)

    conf_name = 'mbox_conf'
    reshape_name = "{}_reshape".format(conf_name)
    reshape_layer = net.layer.add()
    reshape_layer.CopyFrom(L.Reshape(shape=dict(dim=[0, -1, cfg.num_classes])).to_proto().layer[0])
    reshape_layer.name = reshape_name
    reshape_layer.top[0] = reshape_name
    reshape_layer.bottom.append(cfg.conf_layer)

    softmax_name = "{}_softmax".format(conf_name)
    softmax_layer = net.layer.add()
    softmax_layer.CopyFrom(L.Softmax(axis=2).to_proto().layer[0])
    softmax_layer.name = softmax_name
    softmax_layer.top[0] = softmax_name
    softmax_layer.bottom.append(reshape_name)

    flatten_name = "{}_flatten".format(conf_name)
    flatten_layer = net.layer.add()
    flatten_layer.CopyFrom(L.Flatten(axis=1).to_proto().layer[0])
    flatten_layer.name = flatten_name
    flatten_layer.top[0] = flatten_name
    flatten_layer.bottom.append(softmax_name)

    det_out_param = {
        'num_classes': cfg.num_classes,
        'share_location': cfg.share_location,
        'background_label_id': 0,
        'nms_param': {'nms_threshold': 0.45, 'top_k': 200},
        'keep_top_k': 100,
        'confidence_threshold': 0.01,
        'code_type': P.PriorBox.CENTER_SIZE,
        }

    # parameters for evaluating detection results.
    det_eval_param = {
        'num_classes': cfg.num_classes,
        'background_label_id': 0,
        'overlap_threshold': 0.5,
        'evaluate_difficult_gt': False,
        }

    detection_out_name = "detection_out"
    detection_out_layer = net.layer.add()
    detection_out_layer.CopyFrom(L.DetectionOutput(detection_output_param=det_out_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST'))).to_proto().layer[0])
    detection_out_layer.name = detection_out_name
    detection_out_layer.top[0] = detection_out_name
    detection_out_layer.bottom.append(cfg.loc_layer)
    detection_out_layer.bottom.append(flatten_name)
    detection_out_layer.bottom.append("mbox_priorbox")

    with open(args.save_name, 'w') as f:
        f.write(str(net))


def main():
    args = parser.parse_args()
    if args.verbose is True:
        print('Using the specified args:')
        print(args)

    run(args)


if __name__ == '__main__':
    main()

