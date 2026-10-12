# Third-party components

## Segment Anything Model 2

The code in `sam2/` and configurations in `sam2_configs/` derive from
[Meta's SAM2](https://github.com/facebookresearch/sam2).
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Licensed under the Apache License 2.0; see [LICENSE](LICENSE).
Original copyright headers are retained in the source files.

## PySODMetrics

The code in `py_sod_metrics/` derives from
[PySODMetrics](https://github.com/lartpang/PySODMetrics).
Copyright (c) 2020 lartpang. Licensed under the MIT License;
see [py_sod_metrics/LICENSE](py_sod_metrics/LICENSE).
This vendored copy includes a per-image maximum F statistic used for
checkpoint selection, in addition to the F-measure curve.

## NEU RSDDS-AUG data

Ground-truth masks and visualization inputs in `predictions/` come from
NEU RSDDS-AUG. The repository's code license does not grant additional
rights to the underlying dataset. See the dataset paper and its distribution
terms when reusing those files. Predictions in `predictions/` and `ablation/`
are provided as experiment outputs.
