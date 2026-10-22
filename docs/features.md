# 功能说明（每个入口能做什么、怎么调、边界在哪）

本文件回答“系统有哪些功能、每个功能的参数与输出是什么”。架构与实现细节见
[architecture.md](architecture.md)。

## 1. 图形界面：检测工作站

```bash
python -m app.main --checkpoint ckpt/glide_net_6440647.pth \
    --detector runs/detect/detection/runs/neudet_yolov8m/weights/best.pt
# 默认 http://127.0.0.1:7860
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--checkpoint` | `ckpt/glide_net_6440647.pth` | 分割权重；省略且无 `--onnx` 会启动失败 |
| `--onnx` | 无 | 用 ONNX 图替代 PyTorch 权重 |
| `--detector` | 无 | YOLOv8 权重；不传则只有分割与分级，无类别 |
| `--port` | 7860 | 端口 |
| `--device` | `cuda` | `cuda` / `cpu` |

### 1.1 Tab 一：单图检测

| 输入 | 说明 |
|------|------|
| RGB 图像 | 必填，支持上传 / 拖拽 / 剪贴板 |
| 深度图 | 可选；不传则深度通道置零，仍可推理 |
| 掩膜阈值 | 0.2–0.9，默认 0.5；改变它**真的**改变二值化结果与所有下游数值 |
| 启用分类分支 | 关掉后只跑分割，`detection_count = 0`、区域 `label = null` |

输出四块：

1. **四联视图**：RGB / 深度（min-max 拉伸后）/ 概率热力图 / 分级叠加结果；
2. **叠加结果**：半透明掩膜 + 按严重程度着色的框 + 顶部判定横幅；
3. **报告卡**：判定 + 一句处置建议 + 六个 gauge（缺陷区域数、面积占比、最高评分、
   检测框数、严重数、耗时）+ 逐缺陷行（类别·置信度·宽高·面积·评分·等级）；
4. **结构化 JSON**：与批量 CSV、评测脚本完全同构的 `InspectionReport.as_dict()`。

底部状态行显示文件名、缺陷数与后端信息。

### 1.2 Tab 二：批量检测

| 项 | 说明 |
|----|------|
| 输入 | 多选图像（建议 ≤ 200 张）；不支持直接选文件夹（Gradio 限制） |
| 输出 | 表格（图像/判定/缺陷数/面积%/严重/中等/轻微/类别/耗时）+ 前 12 张缩略图 + CSV |
| 容错 | 单张异常只记为 `error` 行，不中断整批 |
| CSV 位置 | 系统临时目录（路径在界面上给出，可直接下载） |

需要**深度图配对**或**递归目录**时用下面的 `app.batch` CLI。

### 1.3 Tab 三：模型指标

四张表实时读取仓库内结果文件，不写死数字：

| 表 | 来源 |
|----|------|
| 与 SOTA 方法对比 | `predictions/metrics.csv`（按 S- measure 排序，12 行） |
| 消融实验 | `ablation/metrics.csv`（4 个变体） |
| 系统级评测 | `docs/assets/system-eval.json`（IoU/Dice/延迟/判定分布） |
| 推理延迟 | `docs/assets/latency.csv`（四后端） |

文件缺失时降级为提示文本，不会让界面崩。

### 1.4 Tab 四：系统说明

流水线三个分支的职责、分级公式与阈值来源、四条复现命令、以及“哪些是论文成果、
哪些是实践新增”的边界声明。

## 2. 命令行工具

| 命令 | 功能 | 关键参数 | 输出 |
|------|------|----------|------|
| `python -m app.batch` | 文件夹级批量检测（支持深度配对、递归不需要） | `--images --depths --out --limit --checkpoint/--onnx --detector --threshold --no_detector --no_overlay` | `overlays/`、`reports/`、`batch.csv`、`summary.json` |
| `python test.py` | 批量推理并保存预测 PNG | `--data_path --checkpoint --save_path --limit` | `results/pred/*.png` |
| `python eval.py` | 对预测目录算 SOD 指标 | `--pred_path --gt_path` | 终端表格 |
| `python tools/evaluate_predictions.py` | 多方法批量评测 | `--pred_root --gt_dir --exclude --output` | `metrics.csv` |
| `python tools/eval_system.py` | 端到端系统评测 | `--images --depths --gt --checkpoint/--onnx --detector --limit --threshold` | `system-eval.json` + `system-eval.csv` |
| `python deploy/export_onnx.py` | 导出 ONNX | `--checkpoint --out_dir --size --half --opset` | `*.onnx` |
| `python deploy/benchmark.py` | 四后端延迟 + 保真度 | `--checkpoint --onnx_fp32 --onnx_fp16 --images --depths --limit` | `latency.csv` + `latency.json` |
| `python detection/scripts/train_yolo.py` | 训练 YOLOv8 | `--data --model --epochs --imgsz --batch --workers --project --name --resume` | `runs/detect/<project>/<name>/` |
| `python detection/scripts/report_val.py` | 检测验证指标 | `--weights --data --split --imgsz` | `detection-metrics.json` |
| `python tools/download_neudet.py` | 下载 NEU-DET 并整理为 YOLO 结构 | `--out` | 数据目录 |
| `python tools/make_diagrams.py` | 生成架构图 | `--export --width --out` | `.drawio` + `.png` |
| `python tools/make_charts.py` | 生成报告图表 | `--only training\|comparison\|ablation\|latency\|severity\|gallery` | `docs/assets/*.png` |
| `python tools/make_pr_curve.py` | 从掩膜重算 PR 曲线 | `--methods --out --csv` | `pr-curves.png` + `pr-metrics.csv` |
| `python tools/make_examples.py` | 渲染真实流水线的样例图 | `--checkpoint --detector --limit --scale` | `example-panel.png`、`example-overlay.png` |
| `python -m pytest tests -q` | 单元测试 | — | 23 passed |

## 3. 库级接口（给二次开发用）

```python
from app.engines.pipeline import build_pipeline
from app.viz import render_overlay, render_panel

pipeline = build_pipeline(segmenter_ckpt="ckpt/glide_net_6440647.pth",
                          detector_weights="runs/detect/.../best.pt",
                          device="cuda", fp16=True)

report = pipeline.run("image.bmp", "depth.tiff", use_detector=True, threshold=0.5)
payload = report.as_dict()          # dict，可直接 json.dumps
overlay = render_overlay("image.bmp", report)   # PIL.Image
panel = render_panel("image.bmp", "depth.tiff", report)
```

| 对象 | 位置 | 职责 |
|------|------|------|
| `SegmentationEngine` | `app/engines/segmentation.py` | `predict(rgb, depth, threshold)` → `SegmentationResult(prob, mask, latency_ms, backend, size)` |
| `DetectionEngine` | `app/engines/detection.py` | `predict(rgb_uint8)` → `DetectionResult(detections, latency_ms)` |
| `InspectionPipeline` | `app/engines/pipeline.py` | `run(rgb, depth=None, use_detector=True, threshold=None)` → `InspectionReport` |
| `InspectionReport` | 同上 | `.prob .mask .detections .regions .summary .timings` + `.as_dict()` |

### 3.1 `summary` 字段（UI/CSV/评测共用）

| 字段 | 类型 | 含义 |
|------|------|------|
| `verdict` | `none\|light\|medium\|severe` | 图像级判定（取最严重区域） |
| `defect_count` | int | 连通域数量（已过滤 <0.02% 的噪点） |
| `detection_count` | int | 检测框数量 |
| `severity_counts` | `{light, medium, severe}` | 三档区域数 |
| `classes` | `{中文类别: 数量}` | 继承自检测框的类别统计 |
| `defect_area_ratio` | float | 掩膜像素占比（0–1） |
| `worst_score` | float | 最高区域评分（0–100） |

### 3.2 `regions[]` 字段

`x y w h`（像素框）、`area_px area_ratio`、`elongation`、`mean_conf`、
`score`、`severity`、`label`（可能为 `null`）、`label_conf`、
`components{sqrt_area, linearity}`。

## 4. 错误与边界处理

| 场景 | 行为 |
|------|------|
| 未上传 RGB | 抛 `gr.Error("请先上传待检 RGB 图像。")`，界面弹提示，不写日志 |
| 无深度图 | 深度通道置零，正常输出；不警告（深度是可选项） |
| 模型未加载 | `gr.Error("模型尚未加载，请检查启动参数。")` |
| 批量中单张损坏 | 该行 `status=error` + `error=repr(exc)`，其余继续 |
| 空掩膜 | `verdict="none"`、`regions=[]`、报告卡显示“合格”，不抛异常 |
| 检测器缺失 | `detection_count=0`，区域 `label=null`，界面显示“未分类缺陷” |
| 非 2D 掩膜 | `ValueError("binary mask must be 2-D")` |
| 深度图为 `I;16` | 自动 min-max 拉伸到 0–255 |
| 概率图全常数 | `normalise_prob` 加 `1e-8` 防除零 |
| 指标文件缺失 | 指标 Tab 降级为提示文本，不崩 |

## 5. 明确不做的事

1. **不做视频流/相机接入**：当前只处理静态图像，实时采集需要另加驱动与队列层。
2. **不做缺陷跟踪与去重**：同一缺陷在多张图出现会被各算一次。
3. **不做模型热更新**：换权重需重启服务（Gradio 的 `STATE` 在启动时构建）。
4. **不支持多 GPU**：`SegmentationEngine` 单卡单例。
5. **不提供鉴权**：工作站面向课堂演示，`launch()` 未启用 auth。
   若要部署到公网，必须先加认证 + 限流 + 上传体积限制。
6. **不给分级的置信区间**：分级是规则映射，没有概率语义，无法给出“95% 置信”这类说法。
