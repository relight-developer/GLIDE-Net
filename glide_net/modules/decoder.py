"""
MSDEF decoder for GLIDE-Net.
"""
import torch.nn as nn
import torch.nn.functional as F

from .basic_blocks import DoubleConv
from .msdef import MultiScaleDetailEnhanceFusion


class MSDEFDecoder(nn.Module):
    """Multi-Scale Detail Enhancement and Fusion (MSDEF) decoder."""

    def __init__(self, encoder_channels):
        super().__init__()
        self.encoder_channels = encoder_channels

        # Stage 1 integrates x3.
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.msdef1 = MultiScaleDetailEnhanceFusion(
            encoder_channels, decoder_channels=encoder_channels[3], level_index=2)
        self.conv1 = DoubleConv(encoder_channels[2] + encoder_channels[2] // 2, 64)

        # Stage 2 integrates x2.
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.msdef2 = MultiScaleDetailEnhanceFusion(
            encoder_channels, decoder_channels=64, level_index=1)
        self.conv2 = DoubleConv(encoder_channels[1] + encoder_channels[1] // 2, 64)

        # Stage 3 integrates x1.
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.msdef3 = MultiScaleDetailEnhanceFusion(
            encoder_channels, decoder_channels=64, level_index=0)
        self.conv3 = DoubleConv(encoder_channels[0] + encoder_channels[0] // 2, 64)

        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, encoder_features):
        """encoder_features: [x1, x2, x3, x4] from the encoder."""
        d1_up = self.up1(encoder_features[3])
        x = self.conv1(self.msdef1(encoder_features, d1_up))
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear',
                             align_corners=True)

        d2_up = self.up2(x)
        x = self.conv2(self.msdef2(encoder_features, d2_up))
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear',
                             align_corners=True)

        d3_up = self.up3(x)
        x = self.conv3(self.msdef3(encoder_features, d3_up))
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear',
                            align_corners=True)

        return out, out1, out2
