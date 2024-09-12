import torch
import torch.nn as nn

def make_layer(block, num_blocks, **kwarg):
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64):  # , groups=10, res_scale=1.0, 
        super().__init__()
        # self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        # self.act = nn.ReLU(inplace=True)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.norm = nn.GroupNorm(groups, mid_channels)

    def forward(self, x):
        identity = x  # (n, c, h, w)
        # out = self.relu(self.norm(self.conv1(x)))
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return identity + out   # * self.res_scale


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        # main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)