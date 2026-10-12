"""
Transformer--CNN hybrid encoder for GLIDE-Net.
"""
import torch.nn as nn
import torch.nn.functional as F

from .basic_blocks import SimpleConvNeXtBlock, Adapter


class CNNEncoder(nn.Module):
    """
    CNN branch producing features at three scales, with channel counts chosen to
    match the transformer levels (stage1 -> 144, stage2 -> 288, stage3 -> 576).
    """

    def __init__(self, in_ch=3, out_channels=(144, 288, 576)):
        super().__init__()
        ch1, ch2, ch3 = out_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            SimpleConvNeXtBlock(ch1),
            SimpleConvNeXtBlock(ch1),
        )
        self.ds1 = nn.Conv2d(ch1, ch2, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage2 = nn.Sequential(
            SimpleConvNeXtBlock(ch2),
            SimpleConvNeXtBlock(ch2),
        )
        self.ds2 = nn.Conv2d(ch2, ch3, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage3 = nn.Sequential(
            SimpleConvNeXtBlock(ch3),
            SimpleConvNeXtBlock(ch3),
        )

    def forward(self, x):
        x = self.stem(x)
        r1 = self.stage1(x)
        x = self.ds1(r1)
        r2 = self.stage2(x)
        x = self.ds2(r2)
        r3 = self.stage3(x)
        return r1, r2, r3


class HybridEncoder(nn.Module):
    """Hybrid encoder: frozen SAM2 Hiera plus trainable RGB and depth CNNs."""

    def __init__(self, sam2_model, device="cuda"):
        super().__init__()

        self.vit_encoder = sam2_model.image_encoder.trunk

        # Freeze the Hiera trunk; only the inserted adapters are trained.
        for param in self.vit_encoder.parameters():
            param.requires_grad = False

        blocks = [Adapter(block) for block in self.vit_encoder.blocks]
        self.vit_encoder.blocks = nn.Sequential(*blocks)

        self.cnn_rgb = CNNEncoder(in_ch=3, out_channels=(144, 288, 576))
        self.cnn_depth = CNNEncoder(in_ch=1, out_channels=(144, 288, 576))

    def forward(self, rgb, depth):
        x1_t, x2_t, x3_t, x4_t = self.vit_encoder(rgb)

        r1, r2, r3 = self.cnn_rgb(rgb)
        d1, d2, d3 = self.cnn_depth(depth)

        def align_to(src, tgt):
            if src.shape[2:] != tgt.shape[2:]:
                return F.interpolate(src, size=tgt.shape[2:], mode='bilinear',
                                     align_corners=True)
            return src

        r1, r2, r3 = align_to(r1, x1_t), align_to(r2, x2_t), align_to(r3, x3_t)
        d1, d2, d3 = align_to(d1, x1_t), align_to(d2, x2_t), align_to(d3, x3_t)

        return {
            'vit_features': [x1_t, x2_t, x3_t, x4_t],
            'cnn_rgb_features': [r1, r2, r3],
            'cnn_depth_features': [d1, d2, d3],
        }
