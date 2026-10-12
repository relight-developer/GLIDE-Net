"""
Basic building blocks for GLIDE-Net.
"""
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Adapter(nn.Module):
    """Bottleneck adapter applied before each frozen SAM2 Hiera block."""

    def __init__(self, blk) -> None:
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU(),
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        return self.block(x + prompt)


class SimpleConvNeXtBlock(nn.Module):
    """Depthwise conv + pointwise expansion + GELU + residual."""

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.pw1 = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = x + residual
        return self.norm(x)
