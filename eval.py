"""
Evaluation entry point for GLIDE-Net.

Computes the standard saliency/segmentation metrics (S-measure, F-measure,
E-measure, MAE and weighted F-measure) from saved prediction masks.

Example:
    python eval.py --pred_path ./results/pred --gt_path /path/to/GT_test
"""
import argparse
import os

import cv2

import py_sod_metrics


class MetricAccumulator:
    """Streaming accumulator for the common SOD metrics.

    Both ``pred`` and ``gt`` must be ``uint8`` arrays with values in [0, 255].
    """

    def __init__(self):
        self.mae = py_sod_metrics.MAE()
        self.smeasure = py_sod_metrics.Smeasure()
        self.emeasure = py_sod_metrics.Emeasure()
        self.fmeasure = py_sod_metrics.Fmeasure()
        self.wfmeasure = py_sod_metrics.WeightedFmeasure()

    def update(self, pred, gt):
        self.mae.step(pred=pred, gt=gt)
        self.smeasure.step(pred=pred, gt=gt)
        self.emeasure.step(pred=pred, gt=gt)
        self.fmeasure.step(pred=pred, gt=gt)
        self.wfmeasure.step(pred=pred, gt=gt)

    def results(self):
        em = self.emeasure.get_results()["em"]
        fm = self.fmeasure.get_results()
        return {
            "S": self.smeasure.get_results()["sm"],
            "maxF": fm["fm"]["curve"].max(),
            "meanMaxF": fm["mf"],
            "adpF": fm["fm"]["adp"],
            "maxE": em["curve"].max(),
            "meanE": em["curve"].mean(),
            "adpE": em["adp"],
            "wF": self.wfmeasure.get_results()["wfm"],
            "MAE": self.mae.get_results()["mae"],
        }


def format_results(results, dataset_name=None):
    lines = []
    if dataset_name:
        lines.append(f"Dataset: {dataset_name}")
    lines += [
        f"S-measure:  {results['S']:.4f}",
        f"max F:      {results['maxF']:.4f}",
        f"mean max F: {results['meanMaxF']:.4f}",
        f"adp F:      {results['adpF']:.4f}",
        f"max E:      {results['maxE']:.4f}",
        f"mean E:     {results['meanE']:.4f}",
        f"adp E:      {results['adpE']:.4f}",
        f"wF:         {results['wF']:.4f}",
        f"MAE:        {results['MAE']:.4f}",
    ]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GLIDE-Net predictions")
    parser.add_argument("--pred_path", type=str, required=True,
                        help="Directory containing predicted PNG masks")
    parser.add_argument("--gt_path", type=str, required=True,
                        help="Directory containing ground-truth masks")
    parser.add_argument("--dataset_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    gt_names = sorted(n for n in os.listdir(args.gt_path) if n.lower().endswith(".png"))
    if not gt_names:
        raise ValueError("No ground-truth PNG masks found.")
    accumulator = MetricAccumulator()
    for i, gt_name in enumerate(gt_names):
        stem = os.path.splitext(gt_name)[0]
        gt = cv2.imread(os.path.join(args.gt_path, gt_name), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise ValueError(f"Cannot read ground-truth mask: {gt_name}")
        pred = cv2.imread(os.path.join(args.pred_path, stem + ".png"), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            raise FileNotFoundError(
                f"Missing prediction for '{gt_name}' in {args.pred_path}")
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        accumulator.update(pred, gt)
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(gt_names)}] processed")

    results = accumulator.results()
    print("\n" + format_results(results, args.dataset_name))


if __name__ == "__main__":
    main()
