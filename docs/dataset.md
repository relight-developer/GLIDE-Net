# 数据集

两套数据各司其职：**NEU RSDDS-AUG 做分割**（有像素级真值），**NEU-DET 做检测**
（有边界框与类别）。二者不混用，因为标注粒度不同。

## 1. NEU RSDDS-AUG · 分割数据集

来源：[官方 RSDDS 仓库](https://github.com/neu-rail-rsdds/rsdds)，课程提供的
`../datasets/NEU-RSDDS-AUG`。

| 项 | 值 |
|----|-----|
| 规模 | 训练 1500 张 + 测试 362 张 |
| RGB | `Image_{split}/*.bmp` |
| 深度 | `Depth_{split}/*.tiff`，**uint16**，典型取值范围 42 000–63 000 |
| 真值 | `GT_{split}/*.png`，单通道，0/255 二值 |
| 分辨率 | 每张不同（如 301×223、174×252、632×874），不做固定裁剪 |
| 缺陷面积占比 | 训练/测试合并统计：均值 0.280，p10 0.096，p50 0.254，p90 0.516 |
| 缺陷细长比 | 包围盒长边/短边，中位数 1.59，p90 2.94 |

配对规则见 `dataset.py::_pair_by_stem`：RGB / 深度 / 真值三者按**文件名主名**严格配对，
主名集合不一致直接抛异常（防止静默错配）。

### 1.1 预处理一致性（关键坑）

`dataset.py::TestDataset` 里深度图的处理是：

```python
depth = F.to_tensor(F.resize(depth, size, interpolation=InterpolationMode.BILINEAR))
```

其中 `depth = Image.open(path).convert("L")`。问题在于 16 位 TIFF 用 Pillow 的
`convert("L")` 会**按 255 截断**，结果几乎全白（实测 `1.tiff`：raw 47571–53641 →
转换后 `unique = [255]`）。也就是说训练/复现时的“深度通道”实际上接近常数，
模型主要靠 RGB 分支学习——这也解释了为什么去掉 CNN 分支（含深度流）只降 0.0042。

**实践阶段的处理**：服务层 `app/engines/preprocess.py::load_gray` 对 `I;16`/`I`/`F`
模式做 min-max 拉伸到 0–255，物理意义正确（近处暗、远处亮），并且**不改变**复现口径：

```
对 8 张测试图对比两种深度输入的掩膜差异：
  IoU 均值 0.9916，概率 MAE 0.0026 → 服务层拉伸不影响论文指标的复现结论
```

复现命令见 `docs/experiment-log.md`“深度图饱和验证”。**结论**：论文指标用原版
预处理复现，服务层用拉伸后的深度（更符合实际相机输入），两者差异已量化并被记录。

> 遗留问题（诚实记录）：若要严格对齐，应在训练阶段也改用拉伸深度重训一次，
> 观察指标是否提升。这是明确的后续工作，不是已完成的结论。

## 2. NEU-DET · 检测数据集

来源：Hugging Face `KeenForgeAI/NEU-DET-corrected`，由 `tools/download_neudet.py`
下载并整理成 YOLO 结构。

| 项 | 值 |
|----|-----|
| 规模 | 1797 张（train 1437 / val 180 / test 180） |
| 图像 | 统一 200×200 灰度热轧带钢表面 |
| 标注 | YOLO 格式 `class cx cy w h`（归一化） |
| 类别 | 0 crazing 龟裂 / 1 inclusion 夹杂 / 2 patches 斑块 / 3 pitted_surface 麻点 / 4 rolled-in_scale 氧化皮压入 / 5 scratches 划伤 |
| 目录 | `images/{train,val,test}`、`labels/{train,val,test}`、`data.yaml` |

```bash
python tools/download_neudet.py --out ../datasets/NEU-DET
```

脚本要点：`data.yaml` 里的 `path: .` 是相对路径，而 Ultralytics 会相对**全局
`datasets_dir`**（默认 `~/datasets`）解析，而不是相对 yaml 文件。因此
`detection/scripts/train_yolo.py::resolve_data_yaml` 会把 `path` 与各 split 全部转成
绝对路径后写到临时 yaml 再交给 Ultralytics。这个坑曾导致训练一开始去 `~/datasets`
找数据并报 `FileNotFoundError`。

### 2.1 类别分布

见 `docs/assets/detection-metrics.json` 的 `per_class` 段（由
`detection/scripts/report_val.py` 生成），以及训练曲线
`docs/assets/training-curves.png`。

## 3. 两套数据集的关系

| | NEU RSDDS-AUG | NEU-DET |
|---|---|---|
| 任务 | 像素级分割（+ 深度） | 边界框检测 |
| 缺陷形态 | 钢轨表面长条划伤、斑块、掉块 | 热轧带钢六类 |
| 在本系统中的角色 | 分割分支的输入与评测基准 | 检测分支的训练数据与类别来源 |
| 是否混训 | 否 | 否 |

前端同时展示两路结果时，类别名称来自 NEU-DET 的六类词表，掩膜位置来自 RSDDS-AUG
的分割模型。这是**有意为之的工程取舍**：手头只有 RSDDS-AUG 有像素真值，而只有
NEU-DET 有类别标注；融合层用 IoU 把两者对齐，并在界面上明确标注“类别来自检测分支”。

## 4. 校验清单

```bash
# 1) 三个子目录文件数一致
python - <<'PY'
from pathlib import Path
r = Path('../datasets/NEU-RSDDS-AUG')
for s in ('train', 'test'):
    print(s, len(list((r/f'Image_{s}').iterdir())),
          len(list((r/f'Depth_{s}').iterdir())),
          len(list((r/f'GT_{s}').iterdir())))
PY
# 期望：train 1500 1500 1500 / test 362 362 362

# 2) 真值必须是二值
# test 集抽样：np.unique 只出现 [0, 255]

# 3) 深度图必须是 uint16
# test 集抽样：dtype uint16，min/max 在 4e4~6e4 量级
```

以上三项检查已经固化为 `tests/test_pipeline.py` 中的
`test_depth_is_stretched_to_uint8_range`（用合成 uint16 图验证拉伸逻辑）。
