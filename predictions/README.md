# Comparison predictions

Per-image predictions of GLIDE-Net and 11 competing RGB-D methods on the
**NEU RSDDS-AUG** test set (362 images).

## Layout

```
predictions/
├── GT/                 ground-truth masks (PNG, 8-bit)
├── Ours/               GLIDE-Net predictions
├── SAM2-Adapter/
├── OpenRSS/
├── RDNet-KD/
├── SAINet/
├── MENet-KD/
├── SA2F/
├── FHENet/
├── RFNet/
├── SPNet/
├── TriTransNet/
├── CIR-Net/
├── inputs/
│   ├── RGB/            input RGB images (JPEG, quality 95)
│   └── Depth/          input depth maps (PNG)
├── metrics.csv         metrics computed from the masks above
└── README.md
```

Prediction and GT folders contain the same 362 PNG file names (`1.png` through
`362.png`) and identical spatial sizes. Visualization RGB files use JPEG;
depth files use PNG. Use the original BMP/TIFF dataset for model inference.

## Metrics

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

`metrics.csv` additionally reports `meanE`, `adpF` and `wF`.

Metric definitions (computed with the vendored `py_sod_metrics`):

- **S-measure** — `Smeasure().get_results()["sm"]`.
- **MAE** — `MAE().get_results()["mae"]`.
- **max-E** — maximum of the averaged E-measure curve, `Emeasure()["em"]["curve"].max()`.
- **max-F** — maximum of the averaged F-measure curve with `beta = 0.3`,
  `Fmeasure(beta=0.3)["fm"]["curve"].max()` (not the mean of per-image maxima).

## Reproducing the metrics

```bash
python tools/evaluate_predictions.py \
    --pred_root predictions \
    --gt_dir predictions/GT \
    --exclude inputs GT \
    --output predictions/metrics.csv
```

## Notes

- The comparison masks were produced with the official implementations of the
  respective methods, trained/evaluated on NEU RSDDS-AUG.
- Please cite the original papers of the compared methods when using these
  predictions. The folder names match the comparison table of the manuscript.
- The input RGB/depth images are redistributed only to make the predictions
  interpretable; they belong to the NEU RSDDS-AUG dataset.
