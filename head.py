import torch
import torch.nn as nn

from backbone import yolo_params, Conv
from neck import Neck, Backbone


class DFL(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch

        self.conv = nn.Conv2d(
            in_channels=ch, out_channels=1, kernel_size=1, bias=False
        ).requires_grad_(False)
        # initialize conv with [0, ..., ch-1]
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = nn.Parameter(x)  # DFL only has ch parameters

    def forward(self, x):
        # x must have num_channels = 4 * ch: x[bs, 4*ch, c]
        b, c, a = x.shape  # c = 4 * ch
        x = x.view(b, 4, self.ch, a).transpose(1, 2)  # [bs, ch, 4, a]

        # take softmax on channel dimension to get distribution probabilities
        x = x.softmax(1)  # [b, ch, 4, a]
        x = self.conv(x)  # [b, 1, 4, a]

        return x.view(b, 4, a)


class Head(nn.Module):
    def __init__(self, version, ch=16, num_classes=80):
        super().__init__()
        self.ch = ch  # DFL channels
        self.coordinates = self.ch * 4  # no. of bbox coordinates
        self.nc = num_classes
        self.no = self.coordinates + self.nc  # no. of outputs per anchor box
        self.stride = torch.zeros(3)  # strides computed during build

        d, w, r = yolo_params(version=version)

        # for bbox
        self.box = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(
                        int(256 * w),
                        self.coordinates,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    Conv(
                        self.coordinates,
                        self.coordinates,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.Conv2d(
                        self.coordinates, self.coordinates, kernel_size=1, stride=1
                    ),
                ),
                nn.Sequential(
                    Conv(
                        int(512 * w),
                        self.coordinates,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    Conv(
                        self.coordinates,
                        self.coordinates,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.Conv2d(
                        self.coordinates, self.coordinates, kernel_size=1, stride=1
                    ),
                ),
                nn.Sequential(
                    Conv(
                        int(512 * w * r),
                        self.coordinates,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    Conv(
                        self.coordinates,
                        self.coordinates,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.Conv2d(
                        self.coordinates, self.coordinates, kernel_size=1, stride=1
                    ),
                ),
            ]
        )

        # for classification
        self.cls = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(int(256 * w), self.nc, kernel_size=3, stride=1, padding=1),
                    Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1),
                ),
                nn.Sequential(
                    Conv(int(512 * w), self.nc, kernel_size=3, stride=1, padding=1),
                    Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1),
                ),
                nn.Sequential(
                    Conv(int(512 * w * r), self.nc, kernel_size=3, stride=1, padding=1),
                    Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1),
                ),
            ]
        )

        # dfl
        self.dfl = DFL()

    def forward(self, x):
        """
        x: output of Neck => List of 3 tensors with different resolution and different channel dim

        x[0]=[bs, ch0, w0, h0], x[1]=[bs, ch1, w1, h1], x[2]=[bs,ch2, w2, h2]
        """
        for i in range(len(self.box)):  # detection head i
            box = self.box[i](x[i])  # [bs,num_coordinates,w,h]
            cls = self.cls[i](x[i])  # [bs,num_classes,w,h]
            x[i] = torch.cat((box, cls), dim=1)  # [bs,num_coordinates+num_classes,w,h]

        if self.training:
            return x  # if in training, no DFL output

        # in inference, DFL produces refined bounding box coordinates
        anchors, strides = (
            i.transpose(0, 1) for i in self.make_anchors(x, self.stride)
        )

        # Concatenate predictions from all detection layers
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        # Split out predictions for box and cls
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)  # a = b = [bs, 2 * self.ch, sum_i(h[i], w[i])]
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * strides, cls.sigmoid()), dim=1)

    def make_anchors(self, x, strides, offset=0.5):
        """
        x = List of feature maps:
            x=[x[0],...,x[N-1]], in our case N= num_detection_heads=3
            each having shape [bs, ch, w, h]
            each feature map x[i] gives output[i] = w*h anchor coordinates + w*h stride values
        strides = List of stride values indicating how much
                   the spatial resolution of the feature map is reduced compared to the original image
        """
        assert x is not None
        anchor_tensor, stride_tensor = [], []
        dtype, device = x[0].dtype, x[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = x[i].shape
            sx = (
                torch.arange(end=w, device=device, dtype=dtype) + offset
            )  # x coordinates of anchor centers
            sy = (
                torch.arange(end=h, device=device, dtype=dtype) + offset
            )  # y coordinates of anchor centers
            sy, sx = torch.meshgrid(sy, sx)  # all anchor centers
            anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(
                torch.full((h * w, 1), stride, dtype=dtype, device=device)
            )

        return torch.cat(anchor_tensor), torch.cat(stride_tensor)


if __name__ == "__main__":
    x = torch.rand((1, 64, 128))

    # # Test DFL
    # dfl = DFL()
    # print(f"{sum(p.numel() for p in dfl.parameters())} parameters")

    # output = dfl(x)
    # print(output)
    # print(dfl)

    # # Test Head
    # neck = Neck(version="n")
    # print(f"{sum(p.numel() for p in neck.parameters()) / 1e6} million parameters")

    # x = torch.rand((1, 3, 640, 640))
    # out1, out2, out3 = Backbone(version="n")(x)
    # out_1, out_2, out_3 = neck(out1, out2, out3)
    # print(out_1.shape)
    # print(out_2.shape)
    # print(out_3.shape)

    # detect = Head(version="n")
    # print(f"{sum(p.numel() for p in detect.parameters())/1e6} million parameters")
    # # out_1,out_2,out_3 are output of the neck
    # output = detect([out_1, out_2, out_3])
    # print(output[0].shape)
    # print(output[1].shape)
    # print(output[2].shape)

    # print(detect)
