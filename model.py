import torch
import torch.nn as nn


class Conv(nn.Module):
    """
    Conv block consisting of Conv2D, BatchNorm2d, SiLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    Bottleneck block: A stack of 2 Conv with shortcut connections
    """

    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut

    def forward(self, x):
        x_in = x  # Residual connections
        x = self.conv1(x)
        x = self.conv2(x)
        if self.shortcut:
            x = x + x_in

        return x


class C2f(nn.Module):
    """
    C2f: Conv + Bottleneck * N + Conv
    """

    def __init__(self, in_channels, out_channels, num_bottlenecks, shortcut=True):
        super().__init__()

        self.mid_channels = out_channels // 2
        self.num_bottlenecks = num_bottlenecks
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # Sequence of Bottleneck layers
        self.m = nn.ModuleList([
            Bottleneck(self.mid_channels, self.mid_channels)
            for _ in range(num_bottlenecks)
        ])
        self.conv2 = Conv((num_bottlenecks + 2) * out_channels // 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        # Split x along channel dimension
        x1, x2 = x[:, :x.shape[1] // 2, :, :], x[:, x.shape[1] // 2:, :, :]
        # List of outputs
        outputs = [x1, x2]

        for i in range(self.num_bottlenecks):
            x1 = self.m[i](x1)  # [bs, 0.5c_out, w, h]
            outputs.insert(0, x1)
        outputs = torch.cat(outputs, dim=1)  # [bs, 0.5c_out(n_bottlenecks + 2), w, h]
        out = self.conv2(outputs)

        return out


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        # Concatenate outputs of max pool and feed to conv2
        self.conv2 = Conv(4 * hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # Max pool is applied at 3 different scales
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2, dilation=1, ceil_mode=False)

    def forward(self, x):
        x = self.conv1(x)

        # Apply Maxpooling at different scales
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        # Concatenate
        y = torch.cat([x, y1, y2, y3], dim=1)
        y = self.conv2(y)  # Final conv

        return y


if __name__ == '__main__':
    c2f = C2f(in_channels=64, out_channels=128, num_bottlenecks=2)
    print(f"{sum(p.numel() for p in c2f.parameters()) / 1e6} million parameters")

    dummy_input = torch.rand((1, 64, 512, 512))
    dummy_input = c2f(dummy_input)
    print("Output shape: ", dummy_input.shape)

    sppf = SPPF(in_channels=128, out_channels=512)
    print(f"{sum(p.numel() for p in sppf.parameters()) / 1e6} million parameters")

    dummy_input = sppf(dummy_input)
    print("Output shape: ", dummy_input.shape)
