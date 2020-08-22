from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from . import resnet
from .kpconv.blocks import KPConv


class DeepLab(nn.Module):
    def __init__(self, backbone, classifier):
        super(DeepLab, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):

        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return x


class DeepLabKP(nn.Module):
    def __init__(self, backbone, classifier):
        super(DeepLabKP, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.kpclassifier = KPClassifier(256, 19)
        self.final = nn.Conv2d(256, 19, 1)

    def forward(self, x, px, py, pxyz, pknn):

        x = self.backbone(x)
        x = self.classifier(x)
        x = self.kpclassifier(x, px, py, pxyz, pknn)
        x = self.final(x)
        return x


def resample_grid(predictions, py, px):
    pypx = torch.stack([px, py], dim=3)
    resampled = F.grid_sample(predictions, pypx)

    return resampled


class KPClassifier(nn.Module):
    def __init__(self, in_channels=256, num_classes=19):
        super(KPClassifier, self).__init__()
        self.kpconv = KPConv(
            kernel_size=15,
            p_dim=3,
            in_channels=in_channels,
            out_channels=256,
            KP_extent=1.2,
            radius=0.60,
        )
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x, px, py, pxyz, pknn):
        x = resample_grid(x, py, px)
        res = []
        for i in range(x.shape[0]):
            points = pxyz[i, ...]
            feats = x[i, ...].transpose(0, 2).squeeze()
            feats = self.kpconv(points, points, pknn[i, ...], feats)
            res.append(feats.unsqueeze(2).transpose(0, 2).unsqueeze(2))
        res = torch.cat(res, axis=0)
        res = self.relu(self.bn(res))
        return res


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__()
        self.aspp = ASPP(in_channels, [12, 24, 36])
        self.lat_lv2 = nn.Sequential(
            nn.Conv2d(512, 128, 1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.mix_lv2 = nn.Sequential(
            nn.Conv2d(640, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.lat_lv1 = nn.Sequential(
            nn.Conv2d(256, 64, 1, padding=0, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        )

        self.mix_lv1 = nn.Sequential(
            nn.Conv2d(576, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.2)
        # self.final = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        lv1, lv2, lv3 = x

        res = self.aspp(lv3)

        lv2 = self.lat_lv2(lv2)
        res = F.interpolate(
            res, size=lv2.shape[-2:], mode="bilinear", align_corners=False
        )
        res = torch.cat([lv2, res], dim=1)
        res = self.mix_lv2(res)

        lv1 = self.lat_lv1(lv1)
        res = F.interpolate(
            res, size=lv1.shape[-2:], mode="bilinear", align_corners=False
        )
        res = torch.cat([lv1, res], dim=1)
        res = self.mix_lv1(res)
        res = self.dropout(res)

        return res


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, 512, 1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def resnext101_aspp_kp(num_classes):
    backbone = resnet.resnext101_32x16d_wsl(
        pretrained=False, replace_stride_with_dilation=[False, False, True]
    )
    classifier = DeepLabHead(2048, num_classes)
    return DeepLabKP(backbone, classifier)
