# 团队分工与交付（7 人）

课程实践按**模块边界**切分，而不是按“谁写文档谁写代码”切分：每位成员的产出都对应
仓库里的一段可运行代码或一组可复现的实验，验收时能单独演示、单独追责。

> 说明：仓库中已有的 GLIDE-Net 训练/评测代码与权重，视为小组前期共同完成的工作
> （论文复现阶段），由 A、B、C 三人负责维护；实践阶段新增的检测分支、融合分级、
> 服务与前端、部署与文档，按下列分工推进。

## 分工总表

| 成员 | 角色 | 负责模块（代码入口） | 核心交付物 | 验收证据 |
|------|------|----------------------|------------|----------|
| **A** | 项目负责人 / 模型与实验 | `glide_net/`、`train.py`、`test.py`、`eval.py` | 论文指标复现、消融实验对照、实验排期 | `predictions/metrics.csv` 12 行、`ablation/metrics.csv` 4 行 |
| **B** | 数据工程 | `dataset.py`、`tools/download_neudet.py` | 两套数据集的下载/校验/配对脚本、预处理一致性核对 | `docs/dataset.md` 统计表、`tests/test_pipeline.py::test_depth_is_stretched_to_uint8_range` |
| **C** | 分割分支工程 | `glide_net/modules/{encoder,cmf,msdef,decoder}.py`、`app/engines/segmentation.py` | 模型结构核对、推理封装、Torch/ONNX 双后端 | `docs/architecture.md` 结构表、`deploy/benchmark.py` 保真度 |
| **D** | 检测分支 | `detection/scripts/train_yolo.py`、`app/engines/detection.py` | YOLOv8m 六类缺陷检测模型、训练曲线、验证指标 | `runs/detect/.../results.csv`、`docs/assets/detection-metrics.json` |
| **E** | 融合与分级 | `app/engines/severity.py`、`app/engines/pipeline.py` | 掩膜-检测框融合规则、三级严重程度评分与标定 | `tests/test_pipeline.py`（10 例）、`docs/severity.md` 标定表 |
| **F** | 服务与前端 | `app/main.py`、`app/ui/`、`app/viz.py`、`app/batch.py` | Gradio 检测工作站、叠加图/四联图渲染、批量 CLI | `docs/assets/ui-workstation.png`、`results/batch_100/batch.csv` |
| **G** | 部署、文档与答辩 | `deploy/`、`docs/`、`tools/make_*.py`、`tools/verify.py` | ONNX 导出与基准、文档体系、验收自检脚本、汇报材料 | `docs/assets/latency.json`、`docs/presentation.md`、`python tools/verify.py` |

## 角色详情

### A · 项目负责人 / 模型与实验

**职责**：把握“复现 → 改进 → 落地”的节奏，负责最终指标口径，对外解释每个数字的来源。

- 复现论文四个关键指标：S-measure 0.8718 / max-F 0.9129 / max-E 0.9475 / MAE 0.0501。
- 组织消融对照（去掉 CNN / Transformer / MSDEF / CMF 四个变体）。
- 维护实验排期表，决定“哪些尝试记录但不进入主线”（如 TensorRT、全量微调）。

**答辩要点**：为什么用冻结 SAM2 + 轻量适配器？→ 见 `docs/architecture.md`“设计取舍”。

### B · 数据工程

**职责**：保证“送进网络的张量和训练时完全一致”，这是所有指标可信的前提。

- 编写 `tools/download_neudet.py`（HF 数据集 → YOLO 目录结构）与配对校验。
- 发现并记录一个关键坑：深度图是 `uint16` TIFF，`Image.convert("L")` 会**饱和成纯白**，
  与训练时的张量化路径不同。已在服务层用 min-max 拉伸修正，并量化了对掩膜的影响
  （平均 IoU 0.9916，见 `docs/dataset.md`）。

### C · 分割分支工程

**职责**：模型结构、推理封装与后端一致性。

- 核对编码器通道 `[144, 288, 576, 1152]`、CMF 三处融合、MSDEF 三级解码。
- 封装 `SegmentationEngine`，统一 Torch/ONNX 两种后端，输出同一种 `prob` 语义
  （sigmoid 概率，逐图 min-max 拉伸，与 `glide_net/checkpoint.py::prediction_mask` 对齐）。
- 交付保真度报告：fp16 相对 fp32 的掩膜 IoU 均值 0.981。

### D · 检测分支

**职责**：给“缺陷是什么”这个问题的答案。

- NEU-DET 六类（crazing / inclusion / patches / pitted_surface / rolled-in_scale / scratches）。
- YOLOv8m，768×768，8 batch，150 epochs；训练曲线自动出图（`tools/make_charts.py`）。
- 输出 `Detection` 结构（类别 + 置信度 + 框），供融合层按 IoU 继承。

### E · 融合与分级

**职责**：把两路输出变成一句人能看懂、能反驳的结论。

- 融合规则三条：有掩膜有框 → 继承类别；只有框 → 保留框但 `severity` 由框面积评分；
  有掩膜无框 → 标“未分类缺陷”而不是丢弃。
- 分级公式与阈值标定：用 362 张真值掩膜的面积分布确定参考值 `AREA_REFERENCE = 0.55`
  （≈ p93），使三档分布均衡（流水线实测 102/120/140，见 `docs/severity.md` 标定前后对比）。
- 全部规则集中在 `GRADE_THRESHOLDS` 一处，报告、CSV、前端共用。

### F · 服务与前端

**职责**：让模型“能用”。四个 Tab：单图检测、批量检测、模型指标、系统说明。

- 视觉设计：深色工业仪表盘（钢青底 `#0B0F14` + 安全黄 `#F5C542`），Barlow Condensed /
  Inter / JetBrains Mono 本地字体，报告卡用 gauge 条显示六个量化维度。
- 工程细节：Gradio 6 的 `theme`/`css` 移到 `launch()`；对比度不足的 `--muted-dim`
  从 3.98:1 调到 5.72:1（WCAG AA）。
- 批量 CLI 逐图容错，单张失败只记 CSV 不中断。

### G · 部署、文档与答辩

**职责**：让“实验”变成“可交付”，并把过程留痕。

- ONNX 导出（fp32/fp16，opset 17，动态 batch）+ 四后端基准 + 保真度对照。
- 图表全部脚本化：`tools/make_diagrams.py`（drawio XML → PNG）、`tools/make_charts.py`
  （读取 results.csv / metrics.csv / latency.json）。
- 文档体系 10 篇 + 验收清单 + 汇报提纲，每个数字标注来源。

## 协作机制

| 机制 | 做法 |
|------|------|
| 接口先行 | 先定 `InspectionReport.as_dict()` 的字段，再各自实现；UI/CSV/评测共用 |
| 单一事实来源 | 阈值、类别名、指标口径只在定义它的模块出现一次 |
| 小步提交 | 一次提交只做一件事，提交信息 `<scope>: <what>`，见下方约定 |
| 失败留痕 | 放弃的方案写进 `docs/experiment-log.md`，附现象与证据 |
| 可复现 | 每个数字都能用一条命令重跑；图表由脚本生成，不手改 PNG |

## 提交信息约定

```
feat(app): 增加批量检测 CLI 与 CSV 报告
fix(severity): 面积参考值由 0.12 调整为 0.55，避免三档退化
perf(deploy): 记录 fp16/ONNX/TensorRT 四后端延迟与保真度
docs(team): 补充 7 人分工与验收证据
test(pipeline): 覆盖分级边界与非 2D 掩膜异常
chore(gitignore): 排除 ckpt/ materials/ runs/ deploy/onnx/
```

## 验收口径（课堂演示用）

| 项 | 目标 | 实测 | 证据文件 |
|----|------|------|----------|
| 分割指标复现 | 与论文一致 | S 0.8718 / MAE 0.0501 | `predictions/metrics.csv` |
| 检测精度 | mAP@0.5 ≥ 0.85（目标） | 0.7721（未达标，原因见 evaluation.md §3） | `docs/assets/detection-metrics.json` |
| 单张端到端延迟 | < 300 ms | P50 60.9 ms（362 张） | `docs/assets/system-eval.json` |
| 覆盖缺陷类别 | ≥ 3 类 | 批量 100 张覆盖 5 类（夹杂/斑块/龟裂/麻点/划伤） | `docs/assets/batch-summary.json` |
| 批量 100 张无崩溃 | 0 error | 100 成功 / 0 失败，20.4 s | `docs/assets/batch-summary.json` |
| 单元测试 | 全绿 | 27 passed | `python -m pytest tests -q` |
