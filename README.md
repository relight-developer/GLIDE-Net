# GLIDE-Net

Official implementation of **GLIDE-Net: Global-Local Interaction and Detail
Enhancement Network for RGB-D Rail Surface Defect Detection**.

GLIDE-Net is an RGB-D segmentation network for rail surface defect detection.
It combines a frozen SAM2 Hiera transformer branch with a trainable
ConvNeXt-style CNN branch, fuses the two modalities through a Cross-Modal
Fusion (CMF) module, and reconstructs the mask with a Multi-Scale Detail
Enhancement and Fusion (MSDEF) decoder.

## Architecture

1. **Hybrid-branch encoder**
   - Transformer branch: SAM2 Hiera-L image encoder with lightweight adapters
     (frozen backbone, trainable adapters).
   - CNN branch: two ConvNeXt-style sub-encoders for the RGB and depth inputs.
2. **Cross-Modal Fusion (CMF)** — Channel Enhancement (CE) followed by Spatial
   Enhancement (SE); applied at the first three encoder levels.
3. **Multi-Scale Detail Enhancement and Fusion (MSDEF) decoder** —
   Weight Distributor (WD) and Detail Enhancement Unit (DEU), where the DEU
   contains Spatial Attention (SA), Channel Attention (CA) and Local Feature
   Preservation (LFP).

## Repository structure

```
.
├── dataset.py                     # RGB-D dataset definitions
├── train.py                       # Training
├── test.py                        # Inference (writes prediction PNGs)
├── eval.py                        # Metric computation
├── glide_net/
│   ├── configs/glide_net_config.py
│   ├── models/glide_net.py        # GLIDE-Net
│   └── modules/
│       ├── basic_blocks.py
│       ├── encoder.py             # Hybrid encoder
│       ├── cmf.py                 # Cross-Modal Fusion
│       ├── msdef.py               # MSDEF / WD / DEU
│       └── decoder.py             # MSDEF decoder
├── tools/
│   ├── evaluate_predictions.py    # Batch metric computation
│   └── make_figures.py            # Build qualitative figures
├── predictions/                   # Predictions of Ours + 11 comparison methods
├── ablation/                      # Predictions of the ablation variants
├── figures/                       # Qualitative comparison / ablation figures
├── sam2/                          # Vendored SAM2 (Apache-2.0, Meta)
├── sam2_configs/                  # SAM2 Hydra configs
└── py_sod_metrics/                # Vendored SOD metric library
```

## Downloads

Download the model checkpoint and comparison/ablation predictions
from [Baidu Netdisk](https://pan.baidu.com/s/12bah0FzeoXm9jZSX5zX42A?pwd=l0p4).
Extraction code: `l0p4`.

| Archive | Contents |
|---------|----------|
| `GLIDE-Net-model.zip` | Full GLIDE-Net checkpoint, including the frozen SAM2 backbone |
| `GLIDE-Net-predictions.zip` | GLIDE-Net, 11 comparison methods, 4 ablation variants, GT and metric tables |

Extract the model archive and pass `glide_net_6440647.pth` to `test.py` via
`--checkpoint`. This full checkpoint does not require `--sam2_checkpoint`.
Obtain the dataset from the [official RSDDS repository](https://github.com/neu-rail-rsdds/rsdds)
and use the prepared dataset root as `--data_path`.

## Installation

```bash
conda create -n glide-net python=3.10
conda activate glide-net

# Install a PyTorch build matching your CUDA version, e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

The repository is run from its root directory (so that `sam2` and
`sam2_configs` are importable).

## Data preparation

We evaluate on the **NEU RSDDS-AUG** dataset. See the
[official RSDDS repository](https://github.com/neu-rail-rsdds/rsdds) for dataset
access and information. Arrange the files as follows:

```
<dataset_root>/
├── Image_train/   *.bmp    # RGB images
├── Depth_train/   *.tiff   # depth maps
├── GT_train/      *.png    # binary masks
├── Image_test/
├── Depth_test/
└── GT_test/
```

RGB, depth and mask files must share the same file stem. Extensions default to
`.bmp` / `.tiff` / `.png` and can be overridden when constructing the dataset
classes in `dataset.py`.

## Pre-trained SAM2 checkpoint

Download the SAM2 Hiera-L checkpoint (`sam2_hiera_large.pt`) from the
[official SAM2 repository](https://github.com/facebookresearch/segment-anything-2)
and pass its path via `--sam2_checkpoint`.

## Training

```bash
python train.py \
    --data_path /path/to/dataset \
    --sam2_checkpoint /path/to/sam2_hiera_large.pt \
    --save_dir ./checkpoints \
    --epochs 50 --batch_size 12 --lr 1e-3
```

Checkpoints are written to `--save_dir`: `last.pth` for resuming and
`best.pth` selected by `1 - MAE + meanMaxF + S`, matching the original experiment.
Here `meanMaxF` is the mean of the per-image maximum F scores; it differs from
the averaged-curve `maxF` reported in the paper. Use `--resume last.pth` to
continue training and `--val_split` to choose the validation split. The original
experiment used the test split for checkpoint selection.

The default loss mode reproduces the original experiment's mean BCE plus
weighted IoU. Use `--loss_mode weighted` for pixel-weighted BCE; this changes
the training objective.

## Inference

```bash
python test.py \
    --data_path /path/to/dataset \
    --checkpoint ./checkpoints/best.pth \
    --sam2_checkpoint /path/to/sam2_hiera_large.pt \
    --save_path ./results/pred
```

Full GLIDE-Net checkpoints include the frozen backbone, so
`--sam2_checkpoint` is unnecessary for those files. Training-generated
checkpoints contain trainable weights and buffers and require the same SAM2
checkpoint used during training. Missing trainable weights, unknown keys and
incompatible shapes cause a loading error.

For CPU inference, add `--device cpu --cpu_threads 4 --num_workers 0`.
Use `--limit 3` for a short run. The supported paper input size is 352×352.
After interrupting CPU inference, `--skip_existing` continues in the same output
folder. Only use it with the same checkpoint, input data and inference settings.

## Evaluation

```bash
python eval.py \
    --pred_path ./results/pred \
    --gt_path /path/to/dataset/GT_test
```

The reported metrics are S-measure, max/adaptive F-measure, max/mean/adaptive
E-measure, weighted F-measure and MAE. Following the paper, `max-F` is the
maximum of the averaged F-measure curve (`Fmeasure(beta=0.3)`), and `max-E` is
the maximum of the averaged E-measure curve.

To (re)compute the metrics of all published prediction folders at once:

```bash
python tools/evaluate_predictions.py \
    --pred_root predictions --gt_dir predictions/GT \
    --exclude inputs GT --output predictions/metrics.csv
```

## Results

Quantitative comparison on the NEU RSDDS-AUG test set (362 images):

| Model        | S-measure | MAE    | max-E  | max-F  |
|--------------|-----------|--------|--------|--------|
| **Ours**     | **0.8718**| **0.0501** | **0.9475** | **0.9129** |
| MENet-KD     | 0.8562    | 0.0559 | 0.9390 | 0.9009 |
| RDNet-KD     | 0.8558    | 0.0581 | 0.9355 | 0.8985 |
| OpenRSS      | 0.8556    | 0.0529 | 0.9391 | 0.8917 |
| SAINet       | 0.8514    | 0.0554 | 0.9363 | 0.8975 |
| SA2F         | 0.8452    | 0.0611 | 0.9268 | 0.8839 |
| SPNet        | 0.8448    | 0.0702 | 0.9238 | 0.8866 |
| RFNet        | 0.8428    | 0.0620 | 0.9307 | 0.8956 |
| TriTransNet  | 0.8418    | 0.0635 | 0.9308 | 0.8843 |
| FHENet       | 0.8371    | 0.0634 | 0.9261 | 0.8814 |
| SAM2-Adapter | 0.8267    | 0.0814 | 0.9056 | 0.8616 |
| CIR-Net      | 0.8183    | 0.0841 | 0.9028 | 0.8669 |

Ablation study:

| Variant           | S-measure | MAE    | max-E  | max-F  |
|-------------------|-----------|--------|--------|--------|
| Full model (Ours) | 0.8718    | 0.0501 | 0.9475 | 0.9129 |
| w/o CNN branch    | 0.8676    | 0.0517 | 0.9443 | 0.9116 |
| w/o Transformer   | 0.7096    | 0.1703 | 0.7965 | 0.7211 |
| w/o MSDEF         | 0.8641    | 0.0558 | 0.9412 | 0.9074 |
| w/o CMF           | 0.8662    | 0.0519 | 0.9451 | 0.9094 |

The per-image masks behind these numbers, the metric CSVs and qualitative
figures are provided in [`predictions/`](predictions), [`ablation/`](ablation)
 and [`figures/`](figures).

## Course practice: from checkpoint to inspection workstation

This repository is also the deliverable of a course practice. On top of the
published GLIDE-Net research code it adds a detection branch, a fusion/grading
layer, a Gradio workstation and a deployment benchmark. Every number quoted in
the practice documents is regenerated by a script.

| Added | Path | What it does |
|-------|------|--------------|
| Inspection engines | `app/engines/` | preprocessing, PyTorch/ONNX segmentation backend, YOLOv8 detector, damage grading, end-to-end pipeline |
| Workstation | `app/main.py`, `app/ui/`, `app/viz.py` | four-tab Gradio UI, overlay/quad-view rendering, JSON + CSV reports |
| Batch CLI | `app/batch.py` | folder-level inspection with per-image fault tolerance |
| Detector training | `detection/scripts/` | YOLOv8m on NEU-DET (6 classes), validation report |
| ONNX + benchmark | `deploy/` | fp32/fp16 export, four-backend latency and fp32-fidelity comparison |
| Reproducible figures | `tools/make_diagrams.py`, `tools/make_charts.py`, `tools/make_pr_curve.py` | draw.io architecture diagrams, charts from CSVs, PR curves from masks |
| System evaluation | `tools/eval_system.py` | end-to-end IoU/Dice/latency/verdict statistics |
| Unit tests | `tests/test_pipeline.py` | 14 CPU-only tests, no checkpoint required |

Quick start:

```bash
python -m app.main --checkpoint ckpt/glide_net_6440647.pth     --detector runs/detect/detection/runs/neudet_yolov8m/weights/best.pt
python -m pytest tests -q
```

Documentation index: [`docs/README.md`](docs/README.md) — architecture, dataset,
evaluation, grading rules, deployment, experiment log, team split and a
presentation script.

Measured additions (see `docs/evaluation.md` and `docs/deployment.md`):

| Item | Value |
|------|-------|
| End-to-end latency (352², RTX 4060 Laptop) | 183 ms P50 (Torch fp16) |
| Mask agreement vs paper reference | S 0.8718 / MAE 0.0501 / maxF 0.9129 / maxE 0.9475 |
| fp16 vs fp32 mask IoU | 0.981 mean / 0.964 min |
| Batch run | 100 images, 0 failures, 20.4 s |
| Trainable parameters | 29.91 M of 242.06 M (12.4%) |

## Citation

Citation details will be added after publication.

## Acknowledgements

This repository builds on
[Segment Anything Model 2](https://github.com/facebookresearch/segment-anything-2)
(Meta Platforms, Apache-2.0) and
[py_sod_metrics](https://github.com/lartpang/PySODMetrics) (MIT). Both are
vendored here. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for copyright and license details.

The comparison masks in `predictions/` are the outputs of the corresponding
third-party methods and remain subject to their original licenses. Please cite
the original papers of those methods when using them.

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE).
