"""
Cross-Modal Fusion (CMF) module for GLIDE-Net.
"""
import torch
import torch.nn as nn


class ChannelEnhancement(nn.Module):
    """Channel Enhancement (CE) branch of the CMF module."""

    def __init__(self, in_planes):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes // 2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, depth):
        # CE = sigmoid(MLP(GMP([F_rgb, F_d]))) * F_d
        x = torch.cat((rgb, depth), dim=1)
        out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return depth * self.sigmoid(out)


class SpatialEnhancement(nn.Module):
    """Spatial Enhancement (SE) branch of the CMF module."""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(max_out))


class CrossModalFusion(nn.Module):
    """Cross-Modal Fusion (CMF): channel enhancement followed by spatial enhancement."""

    def __init__(self, in_planes):
        super().__init__()
        self.channel_enhancement = ChannelEnhancement(in_planes)
        self.spatial_enhancement = SpatialEnhancement()

    def forward(self, rgb, depth):
        # CE = sigmoid(MLP(GMP([F_rgb, F_d]))) * F_d
        x_d = self.channel_enhancement(rgb, depth)
        # SE = sigmoid(Conv(Max_c(CE))) * F_d
        sa = self.spatial_enhancement(x_d)
        depth_enhance = depth * sa
        # Fusion = SE + F_rgb
        return depth_enhance + rgb
