"""Gradio inspection workstation for GLIDE-Net.

Run:
    python -m app.main --checkpoint ckpt/glide_net_6440647.pth \
        --detector runs/detect/detection/runs/neudet_yolov8m/weights/best.pt

Four tabs, each answering a question the acceptance criteria ask for:

``单图检测``   upload RGB (+depth) -> mask, defect classes, severity grades
``批量检测``   run a folder of images, download the CSV, watch the throughput
``模型指标``   the paper's metrics and the deployment latency table
``系统说明``   pipeline, thresholds, and how to reproduce every number
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engines.pipeline import InspectionPipeline, build_pipeline  # noqa: E402
from app.ui import report as R  # noqa: E402
from app.ui.styles import css as ui_css  # noqa: E402
from app.viz import render_overlay, render_panel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

STATE: dict = {"pipeline": None, "backend": "—", "device": "cuda"}


# ---------------------------------------------------------------- callbacks
def _pipeline() -> InspectionPipeline:
    if STATE["pipeline"] is None:
        raise gr.Error("模型尚未加载，请检查启动参数。")
    return STATE["pipeline"]


def inspect_one(rgb, depth, threshold: float, use_detector: bool):
    if rgb is None:
        raise gr.Error("请先上传待检 RGB 图像。")
    pipeline = _pipeline()
    image = Image.fromarray(rgb) if isinstance(rgb, np.ndarray) else Image.open(rgb)
    started = time.perf_counter()
    report = pipeline.run(image, depth, use_detector=use_detector, threshold=threshold)
    elapsed = (time.perf_counter() - started) * 1000.0
    panel = render_panel(image, depth, report)
    overlay = render_overlay(image, report)
    sheet = R.report_sheet(report.as_dict(), elapsed_ms=report.timings["segment_ms"],
                           note=f"分割 {report.timings['segment_ms']:.0f} ms · "
                                f"检测 {report.timings['detect_ms']:.0f} ms · "
                                f"端到端 {elapsed:.0f} ms（含可视化渲染）")
    return panel, overlay, sheet, report.as_dict(), R.status_line(
        f"完成检测：{os.path.basename(getattr(image, 'filename', 'upload'))} · "
        f"{report.summary['defect_count']} 处缺陷", "ok")


def inspect_batch(files, threshold: float, use_detector: bool):
    if not files:
        raise gr.Error("请选择至少一张图像（可多选）。")
    pipeline = _pipeline()
    rows, gallery, t0 = [], [], time.perf_counter()
    for path in files:
        try:
            report = pipeline.run(path, None, use_detector=use_detector,
                                  threshold=threshold)
            s = report.summary
            rows.append([os.path.basename(path), s["verdict"], s["defect_count"],
                         f"{s['defect_area_ratio'] * 100:.2f}",
                         s["severity_counts"]["severe"], s["severity_counts"]["medium"],
                         s["severity_counts"]["light"],
                         ";".join(f"{k}:{v}" for k, v in s["classes"].items()),
                         f"{report.timings['total_ms']:.0f}"])
            if len(gallery) < 12:
                gallery.append((render_overlay(path, report), os.path.basename(path)))
        except Exception as exc:  # one bad file must not abort the batch
            rows.append([os.path.basename(path), "error", 0, "0", 0, 0, 0, str(exc), "0"])
    wall = time.perf_counter() - t0
    csv_path = Path(tempfile.gettempdir()) / "glide_net_batch_report.csv"
    try:
        import csv as _csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["image", "verdict", "defect_count", "area_ratio_pct",
                        "severe", "medium", "light", "classes", "total_ms"])
            w.writerows(rows)
        download = str(csv_path)
    except Exception:
        download = None
    mean_ms = np.mean([float(r[8]) for r in rows]) if rows else 0.0
    return rows, gallery, download, R.status_line(
        f"批量完成 {len(rows)} 张 · 平均 {mean_ms:.0f} ms/张 · "
        f"总耗时 {wall:.1f} s", "ok")


# ---------------------------------------------------------------- tabs
def build_ui() -> gr.Blocks:
    backend = STATE["backend"]
    with gr.Blocks(title="钢轨表面缺陷检测工作站", fill_width=True) as demo:
        gr.HTML(R.title_block(backend=backend, device=STATE["device"]))

        with gr.Tabs():
            with gr.Tab("单图检测"):
                with gr.Row():
                    with gr.Column(scale=3, min_width=300):
                        rgb = gr.Image(label="RGB 图像 / RGB", type="numpy",
                                       sources=["upload", "clipboard"], height=300,
                                       elem_classes="pt-frame")
                        depth = gr.Image(label="深度图（可选）/ Depth", type="numpy",
                                         sources=["upload", "clipboard"], height=300,
                                         elem_classes="pt-frame")
                        with gr.Row():
                            threshold = gr.Slider(0.2, 0.9, value=0.5, step=0.05,
                                                  label="掩膜阈值 / Threshold")
                            use_det = gr.Checkbox(value=True, label="启用分类分支")
                        run_btn = gr.Button("开始检测", variant="primary")
                    with gr.Column(scale=5):
                        panel = gr.Image(label="四联视图：RGB · 深度 · 概率 · 结果",
                                         height=340, elem_classes="pt-frame")
                        overlay = gr.Image(label="叠加结果 / Overlay", height=300,
                                           elem_classes="pt-frame")
                status = gr.HTML(R.status_line("就绪，等待上传图像", "ok"))
                with gr.Row():
                    sheet = gr.HTML(R.empty_report())
                    with gr.Column(min_width=260):
                        raw = gr.JSON(label="结构化报告 / JSON", open=False)

            with gr.Tab("批量检测"):
                gr.HTML('<div class="pt-label" style="margin-bottom:8px">'
                        'BATCH INSPECTION · 文件夹级流水线</div>')
                files = gr.File(file_count="multiple", file_types=["image"],
                                label="选择待检图像（可多选，建议 ≤ 200 张）")
                with gr.Row():
                    batch_btn = gr.Button("开始批量检测", variant="primary")
                    batch_dl = gr.File(label="导出 CSV", interactive=False)
                batch_table = gr.Dataframe(
                    headers=["图像", "判定", "缺陷数", "面积%", "严重", "中等", "轻微",
                             "类别", "耗时ms"],
                    label="批量结果 / Batch results", wrap=True)
                batch_gallery = gr.Gallery(label="结果缩略图", columns=4, height=340)
                batch_status = gr.HTML(R.status_line("等待选择图像", "ok"))

            with gr.Tab("模型指标"):
                gr.HTML(_metrics_html())

            with gr.Tab("系统说明"):
                gr.HTML(_prose_html())

        run_btn.click(inspect_one, inputs=[rgb, depth, threshold, use_det],
                      outputs=[panel, overlay, sheet, raw, status])
        batch_btn.click(inspect_batch, inputs=[files, threshold, use_det],
                        outputs=[batch_table, batch_gallery, batch_dl, batch_status])
    return demo


def _metrics_html() -> str:
    """Metrics read from the repository's own CSVs, with a docs fallback."""
    tiles = []
    try:
        import csv as _csv

        pred = ROOT / "predictions" / "metrics.csv"
        if pred.exists():
            rows = list(_csv.DictReader(pred.open(encoding="utf-8")))
            rows.sort(key=lambda r: float(r["S"]), reverse=True)
            tiles.append({
                "title": "与 SOTA 方法对比", "note": "NEU RSDDS-AUG · 362 张测试图",
                "columns": ["方法", "S-measure", "MAE", "max-E", "max-F"],
                "rows": [[r["Model"], r["S"], r["MAE"], r["maxE"], r["maxF"]]
                         for r in rows],
            })
        abl = ROOT / "ablation" / "metrics.csv"
        if abl.exists():
            rows = list(_csv.DictReader(abl.open(encoding="utf-8")))
            tiles.append({
                "title": "消融实验", "note": "四组变体",
                "columns": ["变体", "S-measure", "MAE", "max-E", "max-F"],
                "rows": [[r["Model"], r["S"], r["MAE"], r["maxE"], r["maxF"]]
                         for r in rows],
            })
        sys_eval = ROOT / "docs" / "assets" / "system-eval.json"
        if sys_eval.exists():
            payload = json.loads(sys_eval.read_text(encoding="utf-8"))
            mask = payload.get("mask", {})
            lat = payload.get("latency_ms", {})
            verdicts = payload.get("verdicts", {})
            tiles.append({
                "title": "系统级评测", "note": f"流水线整体 · {payload.get('images')} 张",
                "columns": ["指标", "数值"],
                "rows": [
                    ["掩膜 IoU 均值", f"{mask.get('iou_mean')}"],
                    ["Dice 均值", f"{mask.get('dice_mean')}"],
                    ["Precision / Recall", f"{mask.get('precision_mean')} / {mask.get('recall_mean')}"],
                    ["端到端延迟 P50 / P95", f"{lat.get('total_p50')} / {lat.get('total_p95')} ms"],
                    ["判定分布 严重/中等/轻微",
                     f"{verdicts.get('severe', 0)} / {verdicts.get('medium', 0)} / {verdicts.get('light', 0)}"],
                ],
            })
        lat = ROOT / "docs" / "assets" / "latency.csv"
        if lat.exists():
            rows = list(_csv.DictReader(lat.open(encoding="utf-8")))
            tiles.append({
                "title": "推理延迟", "note": "单张 352×352 · 含前后处理",
                "columns": ["后端", "设备", "端到端 ms", "吞吐 img/s"],
                "rows": [[r["backend"], r["device"], r["end_to_end_ms"], r["fps"]]
                         for r in rows],
            })
    except Exception as exc:  # degrade to prose rather than crash the UI
        return f'<div id="prose">指标文件读取失败：{exc}</div>'
    if not tiles:
        return '<div id="prose">未找到指标文件，请先运行 tools/ 下的评估脚本。</div>'
    return R.metric_tiles(tiles)


def _prose_html() -> str:
    return """
<div id="prose">
<h2>检测流水线</h2>
<ul>
  <li><b>分割分支</b>：GLIDE-Net（SAM2-Hiera 冻结主干 + ConvNeXt 双流 CNN + CMF 融合 +
      MSDEF 解码器），输入 RGB+深度 4 通道 352×352，输出缺陷概率图。</li>
  <li><b>分类分支</b>：YOLOv8m 在 NEU-DET 六类缺陷上训练，给出缺陷类别与包围框。</li>
  <li><b>融合与分级</b>：掩膜连通域按 IoU 继承检测框类别；每个区域用
      面积项 × 细长项 × 置信项得到 0–100 评分，26 / 55 分划分中等 / 严重。</li>
</ul>
<h2>严重程度分级</h2>
<ul>
  <li><code>score = 100 · min(1, area/0.55) · (1 + 0.18 · min(1, (elong-1)/6)) · (0.75 + 0.25 · conf)</code></li>
  <li>面积参考值 0.55 由 362 张真值掩膜标定（面积占比 p25 = 0.15、p75 = 0.39），
      细长比奖励划伤、氧化皮压入一类长条缺陷。</li>
  <li>阈值来源：<code>app/engines/severity.py::GRADE_THRESHOLDS</code>，与报告、CSV 共用同一份定义。</li>
</ul>
<h2>复现命令</h2>
<ul>
  <li><span class="kbd">python -m app.main --checkpoint ckpt/glide_net_6440647.pth --detector &lt;yolo.pt&gt;</span></li>
  <li><span class="kbd">python -m app.batch --onnx deploy/onnx/glide_net_352_fp16.onnx --images ../datasets/NEU-DET/images/test --out results/batch_neudet</span></li>
  <li><span class="kbd">python deploy/benchmark.py --checkpoint ckpt/glide_net_6440647.pth --onnx_fp16 deploy/onnx/glide_net_352_fp16.onnx</span></li>
  <li><span class="kbd">python eval.py --pred_path results/pred --gt_path ../datasets/NEU-RSDDS-AUG/GT_test</span></li>
</ul>
<h2>说明</h2>
<p>本界面为课程实践产物：分割模型与权重来自仓库内已完成的 GLIDE-Net 实验，
检测分支、融合分级、批量流水线、ONNX 部署与前端界面为实践阶段新增。
所有指标均由 <code>py_sod_metrics</code> 在 362 张测试图上重新计算得到。</p>
</div>
"""


def parse_args():
    p = argparse.ArgumentParser(description="GLIDE-Net inspection workstation")
    p.add_argument("--checkpoint", default=str(ROOT / "ckpt" / "glide_net_6440647.pth"))
    p.add_argument("--onnx", default=None)
    p.add_argument("--detector", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument("--no_detector", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.onnx and not os.path.isfile(args.checkpoint):
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    detector = None if args.no_detector else args.detector
    if detector and not os.path.isfile(detector):
        print(f"[warn] detector weights not found ({detector}); 关闭分类分支")
        detector = None

    print("[init] loading segmentation engine ...", flush=True)
    pipeline = build_pipeline(segmenter_ckpt=None if args.onnx else args.checkpoint,
                              onnx_path=args.onnx, detector_weights=detector,
                              device=args.device)
    STATE.update({"pipeline": pipeline, "backend": pipeline.segmenter.name,
                  "device": args.device})
    print(f"[init] backend={pipeline.segmenter.name} "
          f"detector={pipeline.detector.backend if pipeline.detector else 'off'}", flush=True)

    gr.set_static_paths([str(ROOT / "app" / "static")])
    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                css=ui_css(), allowed_paths=[str(ROOT / "app" / "static")],
                quiet=True)


if __name__ == "__main__":
    main()
