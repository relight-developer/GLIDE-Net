"""Fetch the NEU-DET steel-surface dataset used by the detection branch.

The detection branch needs bounding-box labels for six defect classes, which
the segmentation benchmark (NEU RSDDS-AUG) does not provide. NEU-DET is the
same steel-surface domain and ships YOLO-format labels, so it is used for the
classification branch only; the segmentation metrics stay on NEU RSDDS-AUG.

Example:
    python tools/download_neudet.py --dest ../datasets/NEU-DET
"""
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Download NEU-DET (YOLO format)")
    p.add_argument("--dest", default="../datasets/NEU-DET")
    p.add_argument("--repo", default="KeenForgeAI/NEU-DET-corrected")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    from huggingface_hub import snapshot_download

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(args.repo, repo_type="dataset", local_dir=str(dest),
                      max_workers=args.workers)
    counts = {split: len(list((dest / "images" / split).glob("*")))
              for split in ("train", "val", "test") if (dest / "images" / split).is_dir()}
    print(f"NEU-DET ready at {dest}")
    for split, n in counts.items():
        print(f"  {split:<6} {n} images")
    print("classes: crazing, inclusion, patches, pitted_surface, "
          "rolled-in_scale, scratches")


if __name__ == "__main__":
    main()
