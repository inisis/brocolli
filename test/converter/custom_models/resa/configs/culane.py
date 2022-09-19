cfg = dict(
    net=dict(
        type="RESANet",
    ),
    backbone=dict(
        type="ResNetWrapper",
        resnet="resnet50",
        pretrained=True,
        replace_stride_with_dilation=[False, True, True],
        out_conv=True,
        fea_stride=8,
    ),
    resa=dict(
        type="RESA",
        alpha=2.0,
        iter=4,
        input_channel=128,
        conv_stride=9,
    ),
    decoder="PlainDecoder",
    img_height=288,
    img_width=800,
    cut_height=240,
    num_classes=4 + 1,
)
