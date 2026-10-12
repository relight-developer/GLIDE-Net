# Ablation study

Predictions of the four ablation variants on the NEU RSDDS-AUG test set
(362 images), together with the metrics computed from them.

## Layout

```
ablation/
├── w_o_CNN/            dual-branch encoder without the CNN branch
├── w_o_Transformer/    without the SAM2 transformer branch
├── w_o_MSDEF/          without the MSDEF decoder (plain skip connections)
├── w_o_CMF/            without the CMF module
├── metrics.csv
└── README.md
```

Each folder contains the same 362 file names as `predictions/GT`.

## Metrics

| Variant              | S-measure | MAE    | max-E  | max-F  |
|----------------------|-----------|--------|--------|--------|
| Full model (Ours)    | 0.8718    | 0.0501 | 0.9475 | 0.9129 |
| w/o CNN branch       | 0.8676    | 0.0517 | 0.9443 | 0.9116 |
| w/o Transformer      | 0.7096    | 0.1703 | 0.7965 | 0.7211 |
| w/o MSDEF            | 0.8641    | 0.0558 | 0.9412 | 0.9074 |
| w/o CMF              | 0.8662    | 0.0519 | 0.9451 | 0.9094 |

`metrics.csv` additionally reports `meanE`, `adpF` and `wF`. The full-model row
is reproduced here for convenience; its predictions are in `predictions/Ours`.

Metrics use the same definitions as `predictions/README.md`.

## Reproducing the metrics

```bash
python tools/evaluate_predictions.py \
    --pred_root ablation \
    --gt_dir predictions/GT \
    --output ablation/metrics.csv
```
