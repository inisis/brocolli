cfg = dict(
    net=dict(
        type="RESANet",
    ),
    backbone=dict(
        type="ResNetWrapper",
        resnet="resnet34",
        pretrained=True,
        replace_stride_with_dilation=[False, True, True],
        out_conv=True,
        fea_stride=8,
    ),
    resa=dict(
        type="RESA",
        alpha=2.0,
        iter=5,
        input_channel=128,
        conv_stride=9,
    ),
    decoder="BUSD",
    img_height=368,
    img_width=640,
    cut_height=160,
    num_classes=6 + 1,
)
