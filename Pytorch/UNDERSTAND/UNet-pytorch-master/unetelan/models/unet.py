import torch as th
import torch.nn as nn

from torch.nn import functional as F

def noop(x):
    return x


class ConvBlock(nn.Module):
    def __init__(self, in_, out_, pad=0, p=0.5):
        super().__init__()

        self.conv = nn.Conv2d(in_, out_, 3, padding=pad)
        self.bn = nn.BatchNorm2d(out_)
        self.dp = nn.Dropout2d(p=p) if (p > 0) else noop

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dp(x)
        x = F.relu(x)

        return x


class DoubleConv(nn.Module):
    def __init__(self, in_, out_, pad=1, p=0.5):
        super().__init__()

        self.block_1 = ConvBlock(in_, out_, pad, p=p)
        self.block_2 = ConvBlock(out_, out_, pad, p=0.)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)

        return x


class UpConv(nn.Module):
    def __init__(self, in_, out_, pad=1):
        super().__init__()

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(in_, out_, 1, padding=0)

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv(out)

        return out


class UNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self._i = 0
        self.left = []
        self.right = []  # contract and expand paths of the "U" shape

        filter_multipliers = [2 ** k for k in range(cfg.n_layers)]
        self.filter_sizes = [cfg.filters_base * m for m in filter_multipliers]
        self.input_sizes = [cfg.n_channels] + self.filter_sizes

        self.build()

    def _register(self, module, suffix=""):
        setattr(self, f'module_{self._i}{suffix}', module)
        self._i += 1

    def build(self):
        self.pool = nn.MaxPool2d(2, 2)

        for i, (input_size, filter_size) in enumerate(zip(self.input_sizes, self.filter_sizes)):
            module = DoubleConv(input_size, filter_size, self.cfg.pad, self.cfg.dp)

            self.left.append(module)
            self._register(module, "_left")

            upconv = UpConv(2 * filter_size, filter_size, self.cfg.pad)
            module = DoubleConv(2 * filter_size, filter_size, self.cfg.pad, self.cfg.dp)

            if i != len(self.filter_sizes) - 1:
                self.right.append((upconv, module))
                self._register(upconv, "_right_upconv")
                self._register(module, "_right")

            if i == 0:
                self.output_conv = nn.Conv2d(self.filter_sizes[0], self.cfg.n_classes, 1)

    def contract(self, x):
        feat_maps = []

        for i, module in enumerate(self.left):
            input_ = x if (i == 0) else self.pool(feat_maps[-1])

            output_ = module(input_)
            feat_maps.append(output_)

        return feat_maps

    def expand(self, feat_maps):
        output_ = feat_maps[-1]
        for (copy_, (upconv, module)) in reversed(list(zip(feat_maps[:-1], self.right))):
            input_ = upconv(output_)
            input_ = th.cat([input_, copy_], 1)
            output_ = module(input_)

        output_ = self.output_conv(output_)

        return F.sigmoid(output_)

    def forward(self, x):
        out = self.contract(x)
        out = self.expand(out)

        return out