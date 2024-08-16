import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import yolo_params, C2f, Conv, SPPF, Backbone


class Upsample(nn.Module):
    """
    Nearest Neighbor interpolation with scale_factor=2
    """
    def __init__(self, scale_factor=2, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Neck(nn.Module):
    def __init__(self, version):
        super(Neck, self).__init__()
        d, w, r = yolo_params(version)

        self.up = Upsample()
        self.c2f_1 = C2f(in_channels=int(512 * w * (1 + r)), out_channels=int(512 * w), num_bottlenecks=int(3 * d),
                         shortcut=False)
        self.c2f_2 = C2f(in_channels=int(768 * w), out_channels=int(256 * w), num_bottlenecks=int(3 * d),
                         shortcut=False)
        self.c2f_3 = C2f(in_channels=int(768 * w), out_channels=int(512 * w), num_bottlenecks=int(3 * d),
                         shortcut=False)
        self.c2f_4 = C2f(in_channels=int(512 * w * (1 + r)), out_channels=int(512 * w * r), num_bottlenecks=int(3 * d),
                         shortcut=False)

        self.cv_1 = Conv(in_channels=int(256 * w), out_channels=int(256 * w), kernel_size=3, stride=2, padding=1)
        self.cv_2 = Conv(in_channels=int(512 * w), out_channels=int(512 * w), kernel_size=3, stride=2, padding=1)

    def forward(self, x_res_1, x_res_2, x):
        res_1 = x  # residual connection
        x = self.up(x)
        x = torch.cat([x, x_res_2], dim=1)

        res_2 = self.c2f_1(x)  # residual connection
        x = self.up(res_2)
        x = torch.cat([x, x_res_1], dim=1)

        out_1 = self.c2f_2(x)
        x = self.cv_1(out_1)
        x = torch.cat([x, res_2], dim=1)

        out_2 = self.c2f_3(x)
        x = self.cv_2(out_2)
        x = torch.cat([x, res_1], dim=1)
        out_3 = self.c2f_4(x)

        return out_1, out_2, out_3


if __name__ == '__main__':
    neck = Neck(version="n")
    print(f"{sum(p.numel() for p in neck.parameters()) / 1e6} million parameters")

    x = torch.rand((1, 3, 640, 640))
    out1, out2, out3 = Backbone(version='n')(x)
    out_1, out_2, out_3 = neck(out1, out2, out3)
    print(out_1.shape)
    print(out_2.shape)
    print(out_3.shape)
