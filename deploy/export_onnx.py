"""Export GLIDE-Net to ONNX (fp32 and fp16) for the deployment benchmark.

Example:
    python deploy/export_onnx.py --checkpoint ckpt/glide_net_6440647.pth \
        --out_dir deploy/onnx --half
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glide_net.checkpoint import load_model_state, read_checkpoint  # noqa: E402
from glide_net.models.glide_net import GLIDE_Net  # noqa: E402


class _ProbWrapper(torch.nn.Module):
    """Expose sigmoid(main output) so the graph ends with a probability map."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x)[0])


def parse_args():
    p = argparse.ArgumentParser(description="Export GLIDE-Net to ONNX")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sam2_checkpoint", default=None)
    p.add_argument("--out_dir", default="deploy/onnx")
    p.add_argument("--size", type=int, default=352)
    p.add_argument("--half", action="store_true", help="also export an fp16 graph")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = GLIDE_Net(checkpoint_path=args.sam2_checkpoint, device=device).to(device).eval()
    _, state = read_checkpoint(args.checkpoint)
    load_model_state(model, state, allow_frozen_missing=args.sam2_checkpoint is not None)
    del state
    wrapper = _ProbWrapper(model).to(device).eval()

    def export(dtype, path):
        w = wrapper
        x = torch.randn(1, 4, args.size, args.size, device=device, dtype=dtype)
        if dtype == torch.float16:
            w = wrapper.half()
        torch.onnx.export(w, x, path, input_names=["input"], output_names=["mask"],
                          opset_version=args.opset, do_constant_folding=True,
                          dynamic_axes={"input": {0: "batch"}, "mask": {0: "batch"}})
        size_mb = os.path.getsize(path) / 1e6
        print(f"wrote {path} ({size_mb:.0f} MB)")
        return path

    export(torch.float32, os.path.join(args.out_dir, f"glide_net_{args.size}_fp32.onnx"))
    if args.half and device.type == "cuda":
        export(torch.float16, os.path.join(args.out_dir, f"glide_net_{args.size}_fp16.onnx"))


if __name__ == "__main__":
    main()
