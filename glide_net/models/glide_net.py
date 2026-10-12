"""
GLIDE-Net: Global-Local Interaction and Detail Enhancement Network
for RGB-D Rail Surface Defect Detection.
"""
import torch
import torch.nn as nn

from sam2.build_sam import build_sam2

from ..modules.encoder import HybridEncoder
from ..modules.cmf import CrossModalFusion
from ..modules.decoder import MSDEFDecoder


class GLIDE_Net(nn.Module):
    """
    GLIDE-Net: Global-Local Interaction and Detail Enhancement Network.

    Architecture:
        - Hybrid encoder (frozen SAM2 Hiera ViT + trainable CNN)
        - Cross-Modal Fusion (CMF) at the first three encoder levels
        - Multi-Scale Detail Enhancement and Fusion (MSDEF) decoder
    """

    def __init__(self, checkpoint_path=None, device="cuda"):
        super().__init__()

        model_cfg = "sam2_hiera_l.yaml"
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)

        # Only the image encoder trunk is reused; all SAM2 prompt/decoder heads are dropped.
        for name in ["sam_mask_decoder", "sam_prompt_encoder", "memory_encoder",
                     "memory_attention", "mask_downsample", "obj_ptr_tpos_proj",
                     "obj_ptr_proj"]:
            if hasattr(sam2_model, name):
                delattr(sam2_model, name)
        if hasattr(sam2_model.image_encoder, "neck"):
            del sam2_model.image_encoder.neck

        self.encoder = HybridEncoder(sam2_model, device=device)

        self.cmf1 = CrossModalFusion(in_planes=144 + 144)
        self.cmf2 = CrossModalFusion(in_planes=288 + 288)
        self.cmf3 = CrossModalFusion(in_planes=576 + 576)

        encoder_channels = [144, 288, 576, 1152]  # x1, x2, x3, x4
        self.decoder = MSDEFDecoder(encoder_channels)

    def forward(self, x):
        """
        Args:
            x: [B, 4, H, W] (RGB + depth) or [B, 3, H, W] (RGB only, depth set to zero).
        Returns:
            (out, out1, out2): logits at the three decoder scales.
        """
        if x.size(1) == 4:
            rgb = x[:, :3, :, :].contiguous()
            depth = x[:, 3:4, :, :].contiguous()
        elif x.size(1) == 3:
            rgb = x
            depth = torch.zeros(x.size(0), 1, x.size(2), x.size(3),
                                device=x.device, dtype=x.dtype)
        else:
            raise ValueError("Input must have 3 (RGB) or 4 (RGB+D) channels")

        encoder_outputs = self.encoder(rgb, depth)

        vit_features = encoder_outputs['vit_features']        # [x1, x2, x3, x4]
        rgb_features = encoder_outputs['cnn_rgb_features']    # [r1, r2, r3]
        depth_features = encoder_outputs['cnn_depth_features']  # [d1, d2, d3]

        # x_i = CMF(r_i, d_i) + x_i_t
        fusion1 = vit_features[0] + self.cmf1(rgb_features[0], depth_features[0])
        fusion2 = vit_features[1] + self.cmf2(rgb_features[1], depth_features[1])
        fusion3 = vit_features[2] + self.cmf3(rgb_features[2], depth_features[2])

        encoder_features = [fusion1, fusion2, fusion3, vit_features[3]]

        return self.decoder(encoder_features)
