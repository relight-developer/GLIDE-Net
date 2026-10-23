# 实验日志（过程留痕）

按时间顺序记录**做了什么、看到什么、出了什么问题、怎么解决**。失败与弯路同样记录，
因为它们是实践的真实成本。每条都给出可复现的命令或证据。

环境基线：

| 项 | 值 |
|----|-----|
| OS | Windows 11 (10.0.26220) |
| Python | 3.12.13（conda `base`） |
| PyTorch | 2.5.1+cu121 |
| CUDA / cuDNN | 12.1 / 9 |
| GPU | NVIDIA RTX 4060 Laptop，8187 MiB |
| onnxruntime-gpu | 1.22.0 |
| gradio / ultralytics | 6.30.0 / 8.4.183 |

---

## 阶段 0 · 环境

### 0.1 `vpteam` 环境损坏 → 改用 `base`

**现象**：`conda run -n vpteam pip list` 显示 `colorama`、`numpy` 版本为空壳，
`import ssl` 失败，pip 无法联网。

**诊断**：`conda list -n vpteam` 与 `site-packages` 实际内容不一致，环境被部分破坏
（可能是中断的安装）。

**处理**：放弃修复，改用 `base`（Python 3.12 + torch 2.5.1 + CUDA 可用）。
`requirements.txt` 保持原样（面向 Python 3.10 的官方环境），在本机额外补装：

```bash
pip install hydra-core omegaconf gradio onnx onnxruntime-gpu==1.22.0 \
            ultralytics huggingface_hub pytest
```

**教训**：不要在一个坏环境里反复 pip install——先确认 `python -c "import numpy; numpy.__version__"`
能落地。

### 0.2 复现论文指标

```bash
python test.py --data_path ../datasets/NEU-RSDDS-AUG \
    --checkpoint ckpt/glide_net_6440647.pth --save_path results/pred
python eval.py --pred_path results/pred --gt_path ../datasets/NEU-RSDDS-AUG/GT_test
```

结果与论文**四位小数完全一致**：

```
S-measure: 0.8718   max F: 0.9129   mean F: 0.9295   adp F: 0.7910
max E: 0.9475       mean E: 0.9367  adp E: 0.8671    wF: 0.8849
MAE: 0.0501
```

**意义**：权重、预处理、评测脚本三者链条正确，后续任何改动都可以以此为基准做对照。

---

## 阶段 1 · 基准与瓶颈测量

### 1.1 单张前向延迟（352×352，batch 1）

| 配置 | 延迟 |
|------|------|
| fp32 | 220.3 ms（早期测量）/ 179.2 ms（稳定后，见 latency.json） |
| fp16 | 102.7 ms |
| fp16 + channels_last | 101.4 ms |

`channels_last` 在本模型无收益（Hiera 的 attention 不吃 NHWC 布局），放弃。

### 1.2 参数量统计

```python
import sys; sys.path.insert(0, '.')
import torch
from glide_net.models.glide_net import GLIDE_Net
m = GLIDE_Net(checkpoint_path=None, device='cpu')
print(sum(p.numel() for p in m.parameters()) / 1e6)
print(sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6)
```

```
total 242.057721   trainable 29.908425
vit trunk 213.86   adapters 1.71   cnn_rgb 8.96   cnn_depth 8.94
cmf 2.61           decoder 7.68
```

### 1.3 训练显存实测

| 配置 | 峰值显存 | 一步耗时 |
|------|----------|----------|
| bs4, 352², fp32 | 12.26 GB | 14.4 s |

**结论**：8 GB 显存**无法**全量微调。任何架构改动必须保持轻量。
实际尝试 bs4 时进程直接被系统杀掉（`CUDA out of memory` + 后续驱动复位）。

---

## 阶段 2 · 检测分支（YOLOv8m / NEU-DET）

### 2.1 数据准备

```bash
python tools/download_neudet.py --out ../datasets/NEU-DET
```

HF `KeenForgeAI/NEU-DET-corrected`，1797 张、YOLO 标注、6 类。

### 2.2 坑：`data.yaml` 的相对路径

**现象**：训练启动即报 `FileNotFoundError: ~/datasets/images/train` 不存在。

**原因**：Ultralytics 解析 `data.yaml` 里的 `path:` 时，**相对路径以全局
`datasets_dir`（默认 `~/datasets`）为基准**，而不是以 yaml 文件所在目录为基准。

**处理**：写 `detection/scripts/train_yolo.py::resolve_data_yaml()`，把 `path`
与各 split 全部转成绝对路径后 dump 到临时 yaml：

```python
yaml_path = Path(data).resolve()
spec = yaml.safe_load(yaml_path.read_text())
root = Path(spec.get("path") or yaml_path.parent)
if not root.is_absolute():
    root = (yaml_path.parent / root).resolve()
spec["path"] = str(root)
for s in ("train", "val", "test"):
    if s in spec and not Path(spec[s]).is_absolute():
        spec[s] = str((root / spec[s]).resolve())
```

**教训**：第三方库的路径语义一定要读代码验证，不能靠“看起来对”。

### 2.3 训练

```bash
python detection/scripts/train_yolo.py --data ../datasets/NEU-DET/data.yaml \
    --model yolov8m.pt --epochs 150 --imgsz 768 --batch 8 --workers 6 \
    --project detection/runs --name neudet_yolov8m
```

- 实测 **23 s/epoch**（150 epoch 合计 3 468 s ≈ 57.8 min），显存 4.6–4.9 GB。
  早期记录的 70 s/epoch 是首个 epoch 含 dataloader 预热与缓存构建的高值，稳态远低于此。
- 输出实际落在 `runs/detect/detection/runs/neudet_yolov8m/`
  （Ultralytics 会把相对 `project` 拼在 `runs/detect/` 下，**不是** `detection/runs/`）。
  这个路径差异导致过一次 UI 启动失败（找不到 `best.pt`），已在启动命令里用真实路径。
- 指标走向：epoch 17 → mAP@0.5 ≈ 0.58；epoch 45 → 0.62；epoch 60 → 0.69；
  epoch 79 → 0.705。**震荡明显**，原因见下。

### 2.4 现象：mAP 震荡

观察 `results.csv` 的 `metrics/mAP50(B)` 列，在 epoch 40–75 之间在 0.62–0.71 间跳变。

**分析**：

1. NEU-DET 类别极不均衡：`patches`、`pitted_surface` 样本数远少于 `crazing`。
2. 图片仅 200×200 却被放大到 768 训练 → 上采样后纹理模糊，模型在细粒度特征上不稳。
3. `cls_loss` 在学习率峰值阶段反复。

**处理**：保留完整 150 epoch（Ultralytics 的余弦调度后期会收敛），不中途干预；
用 `best.pt` 而非 `last.pt` 做评测与部署。

**中途终止与续训**：训练在 epoch 94 被手动 `taskkill` 终止（要为 TensorRT 基准腾显存）。
之后用 `python detection/scripts/train_yolo.py --resume` 从 `last.pt` 恢复，
Ultralytics 以该 checkpoint 的 `args.yaml` 为准重建优化器/EMA/epoch 状态，
从 95 续跑到 150，曲线连续无断层。

| epoch | mAP@0.5 | mAP@0.5:0.95 |
|-------|---------|--------------|
| 17 | 0.582 | — |
| 45 | 0.621 | — |
| 79 | 0.705 | — |
| 86（终止前 best） | 0.747 | 0.397 |
| 94（终止点） | 0.713 | 0.387 |
| 126（全程 mAP50 最高） | 0.768 | 0.404 |
| **146（最终 best.pt）** | **0.754** | **0.414** |

> 续训带来 +0.025 mAP@0.5（0.747 → 0.772 测试集口径），代价是 40 分钟。
> 这条记录说明：**中断不是灾难**，只要 `last.pt` 在，续训的曲线是连续的。

**未做**：类别重采样、mosaic 关闭实验、更大 imgsz（显存不够）。这些是明确的后续项。

---

## 阶段 3 · 应用层（本实践的主体）

### 3.1 模块拆分

```
app/engines/preprocess.py   图像/深度加载、归一化、概率后处理
app/engines/segmentation.py Torch / ONNX 双后端封装
app/engines/detection.py    YOLOv8 封装 + 六类中英映射
app/engines/severity.py     连通域 + 三级评分（唯一事实来源）
app/engines/pipeline.py     四步流水线 + InspectionReport
app/viz.py                  Pillow 叠加图 / 四联图
app/batch.py                文件夹级批量 CLI
app/ui/{styles,report}.py   设计系统 + 报告卡 HTML
app/main.py                 Gradio 四 Tab 工作站
```

### 3.2 坑：`load_model_state` 签名

`_TorchBackend` 最初调用 `load_model_state(model, state, allow_frozen_missing=True)`
不报错，但如果省略 `sam2_checkpoint`，冻结主干权重缺失会静默留空（模型输出全是噪声）。

**处理**：`SegmentationEngine` 显式把 `sam2_checkpoint` 透传给 `_TorchBackend`，
并在 `allow_frozen_missing` 上用 `sam2_checkpoint is not None` 判断，语义与 `test.py` 一致。

**教训**：`allow_*` 这类开关必须有明确的判断依据，不能默认 True。

### 3.3 坑：深度图 16 位 TIFF 被饱和成纯白

**现象**：`Image.open('Depth_test/1.tiff')` 的 raw 是 `uint16`（47571–53641），
但 `convert("L")` 后 `np.unique()` 只有 `[255]`——整张图纯白。

**影响**：`dataset.py::TestDataset` 走的正是这条路径，即训练/评测时“深度通道”是常数。
这解释了消融中 w/o CNN（含深度流）只降 0.0042。

**量化影响**（服务层拉伸 vs 原版 convert("L")，8 张测试图）：

```
1   depthA uniq [255]  depthB mean 118.6  IoU 0.9837  MAE 0.0036
10  depthA uniq [255]  depthB mean 142.6  IoU 0.9973  MAE 0.0024
100 depthA uniq [255]  depthB mean 143.8  IoU 0.9794  MAE 0.0049
...
mean IoU 0.9916  mean MAE 0.0026
```

**处理**：

- 服务层（`app/engines/preprocess.py::load_gray`）对 `I;16`/`I`/`F` 模式做 min-max 拉伸，
  物理意义正确。
- **论文指标复现仍用原版预处理**，保证与论文可比。
- 在 [dataset.md](dataset.md) 与 [evaluation.md](evaluation.md) 中明确写出这条差异。

**未做**：用拉伸深度重训一遍看指标是否提升。这是最有价值的后续实验，
但本机显存不允许（见 1.3）。

复现脚本：

```python
from PIL import Image
import numpy as np
im = Image.open('../datasets/NEU-RSDDS-AUG/Depth_test/1.tiff')
print(im.mode, np.array(im).min(), np.array(im).max())   # I;16 47571 53641
print(np.unique(np.array(im.convert('L')))[:5])           # [255]  ← 饱和
```

### 3.4 坑：Gradio 6 的 API 变更

| 现象 | 原因 | 处理 |
|------|------|------|
| `Blocks(theme=...)` 报参数已移除 | Gradio 6 把 `theme`/`css` 移到 `launch()` | 改为 `demo.launch(css=..., allowed_paths=[...])` |
| `launch(show_api=...)` 报未知参数 | 6.x 移除 | 删除该参数 |
| `AttributeError: 'str' object has no attribute 'name'` | 自定义 theme 的字体必须传 `gr.themes.GoogleFont` 对象，不能传字符串 | 放弃自定义 theme，只用 `css=` 注入设计系统 |

```python
# 最终形态
demo.launch(server_port=args.port, css=ui_css(),
            allowed_paths=[str(ROOT / "app" / "static")])
```

### 3.5 坑：对比度不足

初次浏览器检查发现 `--muted-dim: #6B7A89` 在 `#0B0F14` 底上对比度 **3.98:1**，
低于 WCAG AA 的 4.5。

**处理**：改为 `#8695A4`（5.72:1），`app/ui/styles.py` 与 `app/ui/report.py` 同步替换
（两处都定义了这个变量，属于重复定义的技术债，已在注释中标明）。

### 3.6 坑：浏览器端文件上传被拦截

**现象**：通过 kimi-webbridge 驱动上传时，Chrome 弹“文件访问权限”，无法自动点击。

**处理**：改用 `gradio_api/upload` + `/call/inspect_one` 的 HTTP 接口做端到端验证
（返回 200 且 JSON 字段正确），浏览器端只验证 UI 渲染与交互元素。
**结论**：上传能力本身正常，限制在浏览器自动化的权限策略，不是应用缺陷。

### 3.7 坑：批量 CSV 写到 `/tmp`

Windows 下 Git-Bash 的 `/tmp` 与 Python 的路径解析不互通，`tempfile` 写到
`C:\Users\...\AppData\Local\Temp`，而脚本用 `/tmp/...` 拼路径找不到。

**处理**：统一用 `tempfile.gettempdir()`，日志重定向也用 `D:/tmp/xxx.log`。

---

## 阶段 4 · 分级标定

### 4.1 问题：三档退化成一档

`tools/eval_system.py` 第一次跑 40 张，`verdicts = {"severe": 40}`——**全部严重**。

**排查**：`area_ratio` 在 0.086–0.714 之间，`worst_score` 65.3–100。
原公式 `AREA_REFERENCE = 0.12` + `sqrt()` → 面积占比 > 0.12 时 `area_term = 1`，
几乎所有样本饱和。

**统计数据集本身的面积分布**（`predictions/GT`，362 张）：

```
p10 0.0964  p25 0.1479  p50 0.2536  p75 0.3867  p90 0.5157
```

**处理**：`AREA_REFERENCE = 0.55`（≈ p93），去掉 `sqrt` 改线性。

**结果**（362 张真值 + 我们的概率图）：

```
逐图判定:  {'light': 108, 'medium': 117, 'severe': 137}
逐区域:    {'light': 250, 'medium': 120, 'severe': 137}
```

三档都有合理占比。流水线全量 362 张（含检测融合）为
`light 102 / medium 120 / severe 140`，同样接近三分。

**补记（后续发现）**：改标定当天只用 `--limit 60` 抽样验证，当时得到
`light 8 / medium 9 / severe 43`（72% 严重），一度担心标定过松。全量 362 张复测后
偏差消失，说明**是小样本抽样的分布问题，不是阈值问题**。教训：参数标定必须用
全量数据验证，抽样只能用于冒烟测试。

**教训**：阈值不能凭直觉定，必须先看数据分布。这也是本实践里唯一一次“公式没错但
参数不合数据”的典型问题。

---

## 阶段 5 · 部署

### 5.1 ONNX 导出

```bash
python deploy/export_onnx.py --checkpoint ckpt/glide_net_6440647.pth \
    --out_dir deploy/onnx --half
```

产物：`glide_net_352_fp32.onnx`（970 MB）、`glide_net_352_fp16.onnx`（486 MB），
opset 17，动态 batch，输出 sigmoid 概率图。

**坑 1**：导出路径用 Git-Bash 的 `/d/...` 形式会触发 `FileNotFoundError`（torch 内部
按 Windows 路径解析）。**处理**：脚本内统一用相对路径。

**坑 2**：ORT 加载时报
`Error merging shape info for output '/m/encoder/vit_encoder/Concat_3_output_0' source:{4} target:{5}`。
**分析**：动态 batch 与 Hiera 的 `Resize`/`Tile` 形状推断冲突，仅是告警，
fp32 一致性仍达 MAE 2.7e-05 / IoU 0.99989。**处理**：接受告警，记录在案。

### 5.2 ORT 的 DLL 问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `onnxruntime_providers_tensorrt.dll` 依赖 `cublasLt64_13.dll` 缺失 | ORT 1.31 要求 CUDA 13 | 降级到 1.22.0（CUDA 12） |
| ORT CUDA EP 不可用 | 需要 torch 的 CUDA DLL 在搜索路径 | `os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), 'lib'))` |
| `nvinfer_10.dll` 缺失 | TensorRT DLL 不在搜索路径 | `pip install tensorrt-cu12==10.9.0.34 tensorrt-cu12-libs==10.9.0.34` + 注册目录 |

### 5.3 TensorRT：未完成

provider 注册成功后，引擎构建未在 10 分钟内完成；同时 YOLO 训练占满 GPU
（4.9 GB / 100%），Windows 页面文件可用量降到 876 MB，进程被拒绝分配内存：

```
OSError(22, '页面文件太小，无法完成操作。', None, 1455)
```

**处理**：终止该尝试，**不记录任何 TensorRT 数字**。在
[deployment.md §3](deployment.md#3-tensorrt尝试与失败记录) 中完整记录失败原因与
“换环境后该怎么做”。

**教训**：并行的重型任务（训练 + 引擎构建 + 多个 Python 进程）在 8 GB 笔记本上会互相
踩踏。查证工具：`systeminfo | grep "Virtual Memory: Available"`、
`Get-Process | Sort PrivateMemorySize64`。

### 5.4 最终后端选择

**Torch fp16，79.43 ms**。理由与代价对照见
[deployment.md §1–2](deployment.md)。ONNX 定位为“可移植”而非“加速”。

---

## 阶段 6 · 文档与图表工程化

### 6.1 图表全部脚本化

| 脚本 | 产物 | 数据来源 |
|------|------|----------|
| `tools/make_diagrams.py` | `docs/assets/*.drawio` + `*.png` | 代码里的 `MODEL_DIAGRAM` 等结构定义 |
| `tools/make_charts.py` | 训练曲线、对比柱状、消融、延迟、评分分布、样例画廊 | `results.csv`、`metrics.csv`、`latency.json`、`system-eval.csv` |
| `tools/make_pr_curve.py` | PR 曲线 + `pr-metrics.csv` | `predictions/` 下的掩膜 |

**为什么**：手改 PNG 一定会和正文漂移；脚本化后改数据即改图。

### 6.2 坑：matplotlib 中文乱码

默认 `DejaVu Sans` 无中文 → 方块。**处理**：

```python
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
```

### 6.3 坑：`py_sod_metrics` 的 PR 曲线接口

`get_results()` 返回的 `["fm"]["curve"]` 只是 F-measure 的 256 个点；
精度/召回在 `["pr"]["p"]` 与 `["pr"]["r"]`（并行数组）。第一次照着论文的写法取
`curve["precision"]` 直接 `IndexError`。

另一个坑：**`gt` 必须传 0/255**，因为 `prepare_data` 内部按 `gt > 128` 二值化；
我一开始传 `gt > 127` 的 0/1 数组，导致 `FG = 0`，所有指标变 0，PR 曲线空白。
修好后 `Ours` 的 maxF = 0.9129，与 `metrics.csv` 一致（互为交叉验证）。

### 6.4 坑：AP 的两种算法差很多

直接 `np.trapezoid(precision, recall)` 得到 `MENet-KD AP = 0.2638` 这种荒谬值，
因为曲线在 recall 极小处 precision 抖动剧烈，且阈值 0 处 recall = 1、precision = 0.28。

**处理**：丢掉 `recall == 0` 的点，对 precision 取**单调包络**
（`np.maximum.accumulate` 逆序），再积分。结果 `Ours 0.9464 / SPNet 0.9451`。

**诚实记录**：AP 上 SPNet 与我们只差 0.0013，而 S-measure 差 0.027。
这说明指标选择会改变叙事，报告里必须两个指标都给，并说明为什么不以 AP 为主。

---

## 阶段 7 · 收尾

### 7.1 `.gitignore` 规范化

大文件与课程材料一律不进仓库：

```
ckpt/          # 924 MB 权重
materials/     # 论文稿件与原始插图
runs/          # YOLO 训练产物 ~500 MB
deploy/onnx/   # ONNX + TRT 缓存 1.4 GB
*.onnx *.pt *.pth *.engine
results/       # 批量输出
```

验证：`git status --short` 中不出现以上路径。

### 7.2 单元测试

```bash
python -m pytest tests -q     # 27 passed
```

CPU-only、无 checkpoint 依赖，评审机器可直接跑。

---

## 附：问题清单速查

| # | 问题 | 类型 | 状态 |
|---|------|------|------|
| 1 | `vpteam` conda 环境损坏 | 环境 | 绕过（用 base） |
| 2 | `data.yaml` 相对路径按 `datasets_dir` 解析 | 库语义 | 修复 |
| 3 | YOLO 输出目录是 `runs/detect/<project>/<name>` | 库语义 | 记录 + 用真实路径 |
| 4 | 16 位深度 TIFF 被 `convert("L")` 饱和 | 数据 | 服务层修复 + 量化影响 |
| 5 | bs4 训练峰值 12.26 GB，8 GB 装不下 | 硬件 | 只训轻量模块 |
| 6 | Gradio 6 `theme`/`css` 位置变更 | 库版本 | 适配 |
| 7 | 自定义 theme 传字符串字体崩 | 库版本 | 改用 `css=` |
| 8 | `--muted-dim` 对比度 3.98 < 4.5 | 可访问性 | 修复到 5.72 |
| 9 | 浏览器上传被权限拦 | 自动化 | 改用 API 路径验证 |
| 10 | Windows `/tmp` 与 Python 不互通 | 平台 | 用 `tempfile.gettempdir()` |
| 11 | 分级三档退化成一档 | 算法参数 | 重新标定 0.55 |
| 12 | ONNX 导出路径 `/d/...` 失败 | 平台 | 用相对路径 |
| 13 | ORT 1.31 需 CUDA 13 | 依赖 | 降级 1.22.0 |
| 14 | ORT CUDA/TRT DLL 缺失 | 依赖 | 注册 DLL 目录 |
| 15 | TensorRT 引擎构建未完成 | 硬件/内存 | 记录为未完成 |
| 16 | matplotlib 中文乱码 | 字体 | 配置 CJK fallback |
| 17 | `py_sod_metrics` PR 接口与 GT 格式 | 库语义 | 修复（gt 传 0/255） |
| 18 | AP 积分法给出荒谬值 | 算法 | 单调包络 + 丢弃 recall=0 |
| 19 | 页面文件不足导致进程被杀 | 环境 | 串行化重型任务 |
| 20 | 架构图 PNG 只有 100 px 高 | 输出未校验 | XML 转义 + 布局自检 |
| 21 | 延迟数字偏高 2 倍（179→88 ms） | 测量方法 | 加 warm-up + `cuda.synchronize()` |

### 7.3 问题 20：架构图导出成了“一条线”

**现象**：`system-architecture.png` 是 2200×101，1.2 MB 看着像图，实际上是
一条几乎空白的横条。`git` 里几张图已经带着这个缺陷提交过一轮。

**排查**：

```bash
python -c "import xml.etree.ElementTree as ET; ET.parse('docs/assets/system-architecture.drawio')"
# xml.etree.ElementTree.ParseError: not well-formed (invalid token): line 13, column 31
```

`value="<b>NEU RSDDS-AUG</b><br/>..."` —— 标签里的 HTML 标记**直接**写进了
XML 属性。`<` 在 XML 里必须转义，文件从一开始就是非法 XML。draw.io CLI 不报错，
只把能认出的部分画出来，于是导出一张 101 px 高的图，退出码仍然是 0。

**根因**：draw.io 的标签是 HTML（`html=1`），`<b>`/`<br/>` 是**内容**不是标记，
但生成器把内容当成了 XML 直接拼接。渲染器和生成器对同一段字符串的解读不一致。

**修复**（`tools/make_diagrams.py`）：

```python
def esc(text):
    text = text.replace("&#xa;", chr(0))          # 保住真正的换行实体
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace(chr(0), "&#xa;")
```

三处拼接（`box` / `label` / `edge`）全部走 `esc()`，源文件里改回自然的 `<b>`
写法。同时加了两道闸门，让同类问题不可能再悄悄发生：

1. `check_layout()` 导出前校验：XML 必须可解析、盒子不得超出画布、
   任意两盒重叠面积不得超过较小者的 35%；有问题直接 `return 2` 不导出。
2. `tools/verify.py` 把“三个 `.drawio` 是合法 XML”变成验收项。

**修复后**：尺寸从 2200×101 / 91 / 668 变为 2200×996 / 776 / 1319，内容像素
占比 0.088 → 0.417、0.15、0.236（按非背景像素统计），三张图在报告里第一次真正可用。

**教训**：**“命令退出码 0”不等于“产物正确”**。凡是生成二进制产物的脚本，都要对
产物本身做一次结构性断言（尺寸、可解析性、内容占比），否则错误会以“看起来还行”
的方式一路走到答辩现场。

### 7.4 问题 21：延迟数字错了 2 倍

**现象**：`docs/assets/latency.csv` 报 torch fp32 179.20 ms / fp16 79.43 ms，
ONNX 256 / 151 ms。这个量级对 242 M 参数、352² 输入的模型其实偏慢，但因为
“fp16 比 fp32 快”的方向是对的，看起来完全合理，一直没被质疑。

**怎么发现的**：一次全量系统评测（362 张）只用了 175 ms/张的端到端时间，
而其中分割单步就报了 79 ms、检测还要 40 ms。同期 `app.batch` 跑 100 张时
单张 76 ms。**同一台机器、同一个模型，两处数字差了 2 倍**，说明至少有一处是错的。

**根因**（三处叠加）：

1. **没有预热**。第一张图承担 CUDA context（~300 ms）+ cuDNN autotune（~200 ms）
   + fp16 权重转换；`--limit 12` 把这个一次性成本摊到 12 张上，直接抬高一倍。
2. **没有同步**。PyTorch 异步下发 kernel，`time.perf_counter()` 记的是入队时间。
   小 batch 下队列没填满，实际算完的时间被低估/高估都可能。
3. **样本太少**。12 张里一两张落在降频窗口就能让均值漂 10%。
   更隐蔽的是当时训练还在跑，GPU 被占 4.9 GB。

**修复**：

```python
# app/engines/segmentation.py
def synchronise(self) -> None:           # 让 backend 暴露同步点
    if self.device.type == "cuda":
        self.torch.cuda.synchronize()

# deploy/benchmark.py
for i in range(max(0, args.warmup)):     # --warmup 默认 3
    engine.backend.run(samples[i % len(samples)][1])
sync()                                   # 计时前后各同步一次
started = time.perf_counter()
...
```

**修复后**（24 张 + 5 次预热，两次独立重跑）：

| 后端 | 修复前 | 修复后 | 差异 |
|------|--------|--------|------|
| torch fp32 | 179.20 ms | 87.67 / 87.60 ms | −51% |
| torch fp16 | 79.43 ms | 40.23 / 40.11 ms | −49% |
| onnx fp32 | 256.23 ms | 75.62 / 75.53 ms | −70% |
| onnx fp16 | 151.24 ms | 48.84 / 48.49 ms | −68% |

重跑误差 < 1%，数字现在可信。端到端 P50 也从 175.1 ms 降到 60.9 ms。

**连带修正**：ONNX 的相对定位变了——预热后 **onnx fp32 比 torch fp32 快**
（75.6 vs 87.7），但 onnx fp16 仍比 torch fp16 慢（48.8 vs 40.2）。
结论（默认 torch fp16、ONNX 定位为可移植）不变，但依据的差值变了，文档已同步。

**教训**：**同一个指标在两处出现且不一致时，先怀疑测量而不是模型。**
性能数字必须写清"预热几次、同步没同步、样本几张"，否则它不可复现，也就不算数据。
这类错误最危险的地方是**方向正确**——如果 fp16 比 fp32 慢，早就被发现并修掉了。
