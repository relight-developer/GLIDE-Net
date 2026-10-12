"""
Multi-Scale Detail Enhancement and Fusion (MSDEF) module for GLIDE-Net.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """Spatial Attention (SA) branch of the DEU."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # SA = sigmoid(Conv_{7x7}(Avg_c(x_i)))
        avg_out = torch.mean(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(avg_out))


class ChannelAttention(nn.Module):
    """Channel Attention (CA) branch of the DEU."""

    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # CA = sigmoid(MLP(AvgPool(x_i)))
        return self.sigmoid(self.fc(self.avg_pool(x)))


class LocalFeaturePreservation(nn.Module):
    """Local Feature Preservation (LFP) branch of the DEU."""

    def __init__(self, dim):
        super().__init__()
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv1x1_1 = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.conv1x1_2 = nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual = x
        x = self.conv3x3(x)
        x = self.conv1x1_1(x)
        x = self.act(x)
        x = self.conv1x1_2(x)
        return self.norm(x + residual)


class DetailEnhancementUnit(nn.Module):
    """Detail Enhancement Unit (DEU): SA, CA and LFP in parallel."""

    def __init__(self, dim):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim)
        self.lfp = LocalFeaturePreservation(dim)

    def forward(self, x):
        # x_i * SA * CA + LFP
        return x * self.ca(x) * self.sa(x) + self.lfp(x)


class WeightDistributor(nn.Module):
    """Weight Distributor (WD) for multi-scale feature aggregation."""

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.num_features = len(in_channels_list)

        # 1x1 convs unifying the channel dimension of every incoming feature.
        self.unify_channels = nn.ModuleList(
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            for in_ch in in_channels_list
        )

        self.weight_distributor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * self.num_features, out_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, self.num_features, 1, bias=False),
            nn.Softmax(dim=1),
        )

    def forward(self, features, base_feature_index):
        base_size = features[base_feature_index].shape[2:]

        unified = []
        for i, feat in enumerate(features):
            if feat.shape[2:] != base_size:
                feat = F.interpolate(feat, size=base_size, mode='bilinear',
                                     align_corners=False)
            unified.append(self.unify_channels[i](feat))

        weights = self.weight_distributor(torch.cat(unified, dim=1))  # [B, N, 1, 1]

        aggregated = torch.zeros_like(unified[0])
        for i, feat in enumerate(unified):
            aggregated = aggregated + feat * weights[:, i:i + 1]
        return aggregated


class MultiScaleDetailEnhanceFusion(nn.Module):
    """MSDEF module replacing the standard skip connection at one decoder level."""

    def __init__(self, encoder_channels, decoder_channels, level_index):
        super().__init__()
        self.level_index = level_index
        current_ch = encoder_channels[level_index]

        self.deu = DetailEnhancementUnit(dim=current_ch)

        # Aggregate the neighbouring scales around the current level.
        wd_in_channels = []
        self.wd_indices = []
        if level_index > 0:
            wd_in_channels.append(encoder_channels[level_index - 1])
            self.wd_indices.append(level_index - 1)
        wd_in_channels.append(current_ch)
        self.wd_indices.append(level_index)
        self.wd_base_index = len(wd_in_channels) - 1
        if level_index < len(encoder_channels) - 1:
            wd_in_channels.append(encoder_channels[level_index + 1])
            self.wd_indices.append(level_index + 1)

        self.weight_distributor = WeightDistributor(wd_in_channels, current_ch)
        self.prepare_decoder_conv = nn.Conv2d(decoder_channels, current_ch // 2,
                                              kernel_size=1, bias=False)

    def forward(self, all_encoder_features, decoder_feature):
        current_encoder_feature = all_encoder_features[self.level_index]

        diff_y = current_encoder_feature.size(2) - decoder_feature.size(2)
        diff_x = current_encoder_feature.size(3) - decoder_feature.size(3)
        if diff_x != 0 or diff_y != 0:
            decoder_feature = F.pad(decoder_feature,
                                    [diff_x // 2, diff_x - diff_x // 2,
                                     diff_y // 2, diff_y - diff_y // 2])

        enhanced_feature = self.deu(current_encoder_feature)

        features_for_wd = [
            enhanced_feature if idx == self.level_index else all_encoder_features[idx]
            for idx in self.wd_indices
        ]
        multiscale_feature = self.weight_distributor(features_for_wd, self.wd_base_index)

        prepared_decoder_feature = self.prepare_decoder_conv(decoder_feature)
        return torch.cat([multiscale_feature, prepared_decoder_feature], dim=1)
