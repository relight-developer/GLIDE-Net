# 验收清单与自检

一条命令验证整个交付物是否自洽（不需要 GPU、不需要权重、约 3 秒）：

```bash
python tools/verify.py          # 人读清单
python tools/verify.py --json   # 机器可读，退出码 0/1
```

自检脚本核对的是**承诺与事实是否一致**：报告里写的数字必须能在结果文件里读到，
文档里链接的文件必须存在，`git` 历史里不能出现大文件，单测必须全绿。任何一条不满足
就返回非 0 退出码，适合直接放进 CI。

## 0. 自检项数

`tools/verify.py` 目前跑 **40 项**断言：论文指标 8、系统评测 6、部署 4、PR 3、文档 9、忽略规则 2、图表 1、幻灯片 5、单测 2。

## 1. 功能验收

| # | 验收项 | 判定标准 | 证据 | 状态 |
|---|--------|----------|------|------|
| F1 | 单图检测 | 上传 RGB（深度可选）→ 出四联视图 + 叠加图 + 报告卡 + JSON | `docs/assets/example-panel.png`、`example-overlay.png` | ✅ |
| F2 | 阈值可调 | 滑块 0.5→0.8 时面积 20.57%→19.74%、评分 37.3→35.9 | `app/engines/pipeline.py::run(threshold=)` + 实测调用 | ✅ |
| F3 | 类别识别 | 区域继承检测框类别（中文展示） | `docs/assets/batch-summary.json`：5 类 | ✅ |
| F4 | 三档分级 | `light/medium/severe` 均可出现，阈值 26/55 | `docs/assets/system-eval.json` 102/120/140 | ✅ |
| F5 | 批量检测 | 多选文件 → 表格 + 缩略图 + CSV，单张失败不中断 | 2 张 API 实测 + 100 张 CLI 实测 | ✅ |
| F6 | 指标看板 | 四张表从 CSV/JSON 实时渲染 | `app/main.py::_metrics_html` | ✅ |
| F7 | 系统说明页 | 讲清三个分支、分级公式、复现命令 | `app/main.py::_prose_html` | ✅ |
| F8 | 批量 CLI | 支持 `--depths` 配对、`--onnx` 后端、`--no_detector` | `python -m app.batch --help` | ✅ |
| F9 | 无深度可用 | 只传 RGB 也能出结果 | `pipeline.run(rgb, None)` | ✅ |
| F10 | 空掩膜 | 判定 `none`，报告显示“合格”，不抛异常 | `tests/test_ui_report.py` | ✅ |

## 2. 指标验收

| # | 验收项 | 判定标准 | 证据 | 状态 |
|---|--------|----------|------|------|
| M1 | 论文指标复现 | S 0.8718 / maxF 0.9129 / maxE 0.9475 / MAE 0.0501（四位小数一致） | `predictions/metrics.csv` | ✅ |
| M2 | 对比方法齐全 | 12 行（Ours + 11 种） | 同上 | ✅ |
| M3 | 消融齐全 | 4 个变体，去掉 Transformer 最差 | `ablation/metrics.csv` | ✅ |
| M4 | 系统级评测 | 全覆盖 362 张、0 失败、IoU 0.8325 | `docs/assets/system-eval.json` | ✅ |
| M5 | 延迟 < 300 ms | 端到端 P50 60.9 ms | 同上 | ✅ |
| M6 | 批量 100 张无崩溃 | 成功 100 / 失败 0 | `docs/assets/batch-summary.json` | ✅ |
| M7 | 检测分支指标 | mAP@0.5 0.7721（**未达 0.85 目标，如实标注**） | `docs/assets/detection-metrics.json` | ⚠️ 见 §5 |
| M8 | PR 曲线可复算 | 从掩膜重算，Ours maxF 与 metrics.csv 一致 | `docs/assets/pr-metrics.csv` | ✅ |

## 3. 部署验收

| # | 验收项 | 判定标准 | 证据 | 状态 |
|---|--------|----------|------|------|
| D1 | ONNX 导出 | fp32 + fp16 两个图可加载、可推理 | `docs/assets/latency.json` | ✅ |
| D2 | 数值保真 | ONNX fp32 与 torch fp32 的 mask IoU > 0.999 | 同上（0.99989） | ✅ |
| D3 | 半精度可用 | fp16 相对 fp32 的 mask IoU ≥ 0.95 | 同上（最差 0.9638） | ✅ |
| D4 | 后端选择有据 | 最快后端 = torch fp16（79.4 ms），并写明代价 | `docs/deployment.md` §2 | ✅ |
| D5 | ONNX 无 PyTorch 也能跑 | `app.batch --onnx` 实测 6 张成功 | 实测输出 | ✅ |
| D6 | TensorRT | **未完成**，如实记录原因，不编造数字 | `docs/deployment.md` §3 | ⚠️ 见 §5 |

## 4. 工程与文档验收

| # | 验收项 | 判定标准 | 证据 | 状态 |
|---|--------|----------|------|------|
| E1 | 单元测试 | ≥ 20 例全绿 | `pytest tests -q` → 27 passed | ✅ |
| E2 | 图表脚本化 | 所有 PNG 都有生成脚本，改数据即改图 | `tools/make_{diagrams,charts,pr_curve,examples}.py` | ✅ |
| E3 | 架构图可编辑 | `.drawio` 源文件 + PNG 导出 | `docs/assets/*.drawio` | ✅ |
| E4 | 文档无死链 | 内部链接全部可达 | `tools/verify.py::check_docs` | ✅ |
| E5 | 索引完整 | `docs/README.md` 链接到每篇文档 | 同上 | ✅ |
| E6 | 大文件不入库 | `ckpt/`、`materials/`、`runs/`、`deploy/onnx/`、`results/` 全忽略，历史干净 | `git check-ignore` + `git log --name-only` | ✅ |
| E7 | 提交规范 | 每次提交只做一件事，`<scope>: <what>` 中文信息 | `git log --oneline` | ✅ |
| E8 | 过程留痕 | 21 条问题清单 + 失败方案记录 | `docs/experiment-log.md` | ✅ |
| E9 | 分工可追溯 | 7 人 A–G，每人对应模块与证据 | `docs/team.md` | ✅ |
| E10 | 可访问性 | 正文对比度 ≥ 4.5:1，无横向溢出 | `docs/experiment-log.md` §3.5 | ✅ |
| E11 | 幻灯片可重建 | `node tools/make_slides.js` 从结果文件生成 14 页，不手改 | `docs/slides.pptx` | ✅ |
| E12 | 幻灯片版式 | 无文字溢出、无重叠、不越安全边距，且无冒号/破折号/翻案句（回读 pptx 逐框核对） | `tools/check_slides.py` | ✅ |

## 5. 已知未达标项（必须在答辩时主动说明）

| 项 | 现状 | 为什么 | 改进路径 |
|----|------|--------|----------|
| **M7 检测 mAP@0.5** | 0.7721（150 epoch 跑满），低于 0.85 目标 | `crazing` 0.40 / `rolled-in_scale` 0.68 拖累；两个类别都是低对比度细纹理，200×200 放大到 768 后信息不足 | 提到 1024+（需 ≥12 GB 显存）、对这两类过采样/copy-paste、末段关闭 mosaic 微调 |
| **D6 TensorRT** | 未完成 | 8 GB 显存被训练占满 + 页面文件不足，引擎构建被系统拒绝分配内存 | 训练/推理分机、16 GB+ 显存、独享页面文件后用 `trtexec` 离线构建 |
| 两路推理并行 | 未实现 | 串行 P50 已 60.9 ms，满足演示需求 | 两个 CUDA stream，预估 61 → ~45 ms |
| 深度流重训 | 未做 | 显存不足，无法全量微调 | 用拉伸后的深度重训，验证 CMF/CNN 的真实贡献 |
| 分级有监督验证 | 无真值 | 数据集只有二值掩膜，无严重程度标注 | 与产线合作获取人工评级，做 Cohen's κ 一致性检验 |

**口径声明**：以上五项在文档里都标注为“未完成/未达标”，并且**没有**出现在任何
性能对比图中。我们不把“尝试过”写成“实现了”，也不给没有测量支撑的数字。

## 6. 演示前 5 分钟检查

```bash
cd /d/projects/vpteam/GLIDE-Net
conda activate base

# 1) 环境与依赖
python -c "import torch, gradio, ultralytics; print(torch.cuda.is_available())"

# 2) 权重与数据在位
ls ckpt/glide_net_6440647.pth runs/detect/runs/detect/detection/runs/neudet_yolov8m/weights/best.pt

# 3) 自检全绿
python tools/verify.py

# 4) 起服务并预热
python -m app.main --port 7860 --checkpoint ckpt/glide_net_6440647.pth \
    --detector runs/detect/runs/detect/detection/runs/neudet_yolov8m/weights/best.pt
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7860/
```

预期：`verify.py` 末行“全部通过 ✅”，HTTP 返回 200。引擎构造时已 warm-up，
首张业务请求 ~180 ms，第二张起稳定 ~61 ms。

## 7. 一句话总结

交付物的可信度不靠“看起来完整”，而靠**每一条都能被一条命令复核**：
指标来自 CSV、图来自脚本、图源可编辑、历史无大文件、未做的写“未做”。
`tools/verify.py` 就是这个承诺的可执行形式。
