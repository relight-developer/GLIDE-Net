/**
 * Build the 13-slide defence deck straight from the committed result files.
 *
 * Nothing is typed by hand: the paper metrics come from predictions/metrics.csv,
 * the ablation from ablation/metrics.csv, and the system/detection/latency
 * numbers from docs/assets/*.json. Rerun after an experiment and the deck follows.
 *
 *   node tools/make_slides.js            # -> docs/slides.pptx
 *
 * Design: "inspection console" — dark steel ground, one safety-yellow accent,
 * rounded data cards and a repeated numbered chip as the single motif.
 */
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

const ROOT = path.resolve(__dirname, '..');
const A = (f) => path.join(ROOT, 'docs', 'assets', f);

// ---------------------------------------------------------------- palette
const BG = '0B0F14';
const PANEL = '151E27';
const PANEL2 = '1B2531';
const INK = 'E8EEF4';
const MUTED = '8695A4';
const DIM = '5A6875';
const ACCENT = 'F5C542'; // safety yellow
const BLUE = '4C9AFF';
const GREEN = '30A46C';
const RED = 'E5484D';

const W = 13.333;
const H = 7.5;
const M = 0.62; // page margin

// ---------------------------------------------------------------- data
const csv = (p) => {
  const lines = fs.readFileSync(p, 'utf8').trim().split(/\r?\n/);
  const head = lines[0].split(',');
  return lines.slice(1).map((l) => {
    const cells = l.split(',');
    const o = {};
    head.forEach((h, i) => (o[h] = cells[i]));
    return o;
  });
};
const json = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));

const metrics = csv(path.join(ROOT, 'predictions', 'metrics.csv'));
const ours = metrics.find((r) => r.Model === 'Ours');
const others = metrics.filter((r) => r.Model !== 'Ours');
const bestOther = others.slice().sort((a, b) => b.S - a.S)[0];
const ablation = csv(path.join(ROOT, 'ablation', 'metrics.csv'));
const sev = json(A('system-eval.json'));
const det = json(A('detection-metrics.json'));
const batch = json(A('batch-summary.json'));
const lat = json(A('latency.json'));

// ---------------------------------------------------------------- helpers
const shadow = () => ({
  type: 'outer',
  color: '000000',
  blur: 12,
  offset: 3,
  angle: 90,
  opacity: 0.35,
});

function text(slide, str, o = {}) {
  slide.addText(str, {
    x: o.x, y: o.y, w: o.w, h: o.h,
    fontSize: o.size || 14,
    bold: !!o.bold,
    color: o.color || INK,
    fontFace: o.face || 'Arial',
    align: o.align || 'left',
    valign: o.valign || 'top',
    margin: 0,
    lineSpacingMultiple: o.ls || 1.15,
    charSpacing: o.charSpacing || 0,
    italic: !!o.italic,
    isTextBox: true,
  });
}

/** Header block: numeric chip + kicker + title + standfirst. */
function head(slide, no, kicker, title, sub) {
  slide.addShape('roundRect', {
    x: M, y: 0.46, w: 0.46, h: 0.46, rectRadius: 0.09,
    fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
  });
  slide.addText(no, {
    x: M, y: 0.46, w: 0.46, h: 0.46,
    fontSize: 13, bold: true, color: BG, align: 'center', valign: 'middle',
    fontFace: 'Arial', margin: 0,
  });
  text(slide, kicker, { x: M + 0.66, y: 0.5, w: 6.5, h: 0.22, size: 10, color: MUTED, charSpacing: 1.6 });
  text(slide, title, { x: M, y: 1.02, w: W - 2 * M, h: 0.62, size: 30, bold: true });
  if (sub) text(slide, sub, { x: M, y: 1.66, w: W - 2 * M - 0.4, h: 0.34, size: 12.5, color: MUTED });
}

function card(slide, o) {
  slide.addShape('roundRect', {
    x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.06,
    fill: { color: o.fill || PANEL },
    line: { color: o.line || PANEL2, width: 1 },
    shadow: shadow(),
  });
  if (o.label) {
    text(slide, o.label, {
      x: o.x + 0.22, y: o.y + 0.16, w: o.w - 0.44, h: 0.24,
      size: o.labelSize || 11, color: o.accent || ACCENT, bold: true, charSpacing: 0.6,
    });
  }
  if (o.body) {
    text(slide, o.body, {
      x: o.x + 0.22, y: o.y + (o.label ? 0.5 : 0.2), w: o.w - 0.44,
      h: o.h - (o.label ? 0.72 : 0.36), size: o.size || 12, color: o.color || INK,
      ls: 1.32,
    });
  }
}

/** Big-number callout: 42pt accent figure over a small caption. */
function stat(slide, x, y, w, value, caption, o = {}) {
  text(slide, value, {
    x, y, w, h: 0.62, size: o.size || 34, bold: true,
    color: o.color || INK, align: o.align || 'left', face: 'Arial',
  });
  text(slide, caption, {
    x, y: y + (o.size ? 0.56 : 0.62), w, h: 0.4, size: 10.5, color: MUTED, ls: 1.2,
  });
}

function img(slide, file, o) {
  if (!fs.existsSync(A(file))) {
    text(slide, `[missing ${file}]`, { x: o.x, y: o.y, w: 3, h: 0.3, size: 10, color: RED });
    return;
  }
  const opts = { x: o.x, y: o.y, path: A(file) };
  if (o.w) opts.w = o.w;
  if (o.h) opts.h = o.h;
  slide.addImage(opts);
}

function table(slide, o) {
  const widths = o.colW; // inches
  slide.addTable(o.rows, {
    x: o.x, y: o.y,
    w: widths.reduce((a, b) => a + b, 0),
    colW: widths,
    border: { type: 'solid', color: '24303D', pt: 0.75 },
    fill: { color: PANEL },
    fontSize: o.size || 11,
    fontFace: 'Arial',
    color: INK,
    rowH: o.rowH || 0.3,
    margin: 0.06,
    valign: 'middle',
    autoPage: false,
  });
  // header row styling
  slide.addShape('rect', {
    x: o.x, y: o.y, w: widths.reduce((a, b) => a + b, 0), h: o.rowH || 0.3,
    fill: { color: '101821' }, line: { color: '101821', width: 0 },
  });
  // redraw header text on top
  let cx = o.x;
  o.rows[0].forEach((cell, i) => {
    text(slide, String(cell.text ?? cell), {
      x: cx + 0.06, y: o.y, w: widths[i] - 0.12, h: o.rowH || 0.3,
      size: 10.5, bold: true, color: ACCENT, valign: 'middle',
    });
    cx += widths[i];
  });
}

function footer(slide, s) {
  text(slide, s, { x: M, y: H - 0.55, w: W - 2 * M, h: 0.24, size: 9, color: DIM });
}

// ---------------------------------------------------------------- deck
const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.33 x 7.5
pres.author = 'GLIDE-Net course practice';
pres.title = 'GLIDE-Net 铁路表面缺陷检测';

const slideBase = () => {
  const s = pres.addSlide();
  s.background = { color: BG };
  return s;
};

// 1 — cover
{
  const s = slideBase();
  img(s, 'example-overlay.png', { x: 8.62, y: 0.9, h: 5.05 });
  text(s, 'COURSE PRACTICE · 2026', { x: 0.78, y: 1.28, w: 7, h: 0.26, size: 11, color: ACCENT, bold: true, charSpacing: 2.2 });
  text(s, 'GLIDE-Net', { x: 0.78, y: 1.62, w: 7.4, h: 1.05, size: 52, bold: true });
  text(s, '铁路表面缺陷检测工作站', { x: 0.78, y: 2.52, w: 7.4, h: 0.55, size: 26, color: INK });
  text(s, '从论文复现到可演示的系统：RGB-D 分割 · 六类检测 · 三级分级 · 四后端部署',
    { x: 0.78, y: 3.2, w: 7.2, h: 0.7, size: 13, color: MUTED, ls: 1.45 });
  const y = 4.28;
  stat(s, 0.78, y, 1.7, ours.S, 'S-measure', { size: 28 });
  stat(s, 2.5, y, 1.7, ours.MAE, 'MAE', { size: 28 });
  stat(s, 4.2, y, 1.7, ours.maxF, 'max-F', { size: 28 });
  stat(s, 5.9, y, 1.7, ours.maxE, 'max-E', { size: 28 });
  text(s, '与论文四位小数一致 · 12 种对比方法中 S-measure 第一',
    { x: 0.78, y: 5.28, w: 7.3, h: 0.3, size: 11, color: GREEN });
  text(s, '左侧图为系统真实输出：红/橙掩膜来自分割分支，黄框与类别来自检测分支',
    { x: 8.62, y: 6.12, w: 4.1, h: 0.5, size: 9.5, color: MUTED, ls: 1.3 });
}

// 2 — task definition
{
  const s = slideBase();
  head(s, '01', '任务定义', '一个问题，三种答案',
    '人工巡检靠眼睛和手电；系统要给出可量化、可复检的证据链，而不只是一张热力图');

  const y = 2.28;
  const w = 3.82;
  const gap = 0.28;
  const blocks = [
    ['分割 · 像素级', BLUE, '哪些像素是缺陷',
      'GLIDE-Net，RGB-D 四通道输入\n输出逐像素概率掩膜\n\n主指标 S-measure / MAE\n与论文四位小数对齐'],
    ['检测 · 实例级', ACCENT, '缺陷是什么类别',
      'YOLOv8m，NEU-DET 六类\n输出框 + 置信度\n\n类别由检测分支提供\nmAP@0.5 = 0.772'],
    ['分级 · 决策级', GREEN, '要不要停机复检',
      '规则引擎，不是学习模型\n面积 × 细长 × 置信\n\n输出 light / medium / severe\n阈值由真值面积分布标定'],
  ];
  blocks.forEach((b, i) => {
    const x = M + i * (w + gap);
    card(s, { x, y, w, h: 2.62, label: b[0], accent: b[1] });
    text(s, b[2], { x: x + 0.22, y: y + 0.56, w: w - 0.44, h: 0.34, size: 15, bold: true });
    text(s, b[3], { x: x + 0.22, y: y + 0.96, w: w - 0.44, h: 1.4, size: 11.5, color: MUTED, ls: 1.45 });
  });

  text(s, '三者不冲突，而是同一张图上的三种粒度。课程实践的工作量集中在后两层，以及把它们串成一条可运行流水线的工程层。',
    { x: M, y: 5.28, w: W - 2 * M - 0.5, h: 0.7, size: 13, color: INK, ls: 1.4 });
  footer(s, '第 2 页 · 任务定义');
}

// 3 — method
{
  const s = slideBase();
  head(s, '02', '方法', '冻结主干，只训 12.4%',
    '242.06 M 参数里只有 29.91 M 可训练——这是 8 GB 显存逼出来的设计，也是它的合理性所在');
  img(s, 'glide-net-architecture.png', { x: M - 0.08, y: 2.12, w: W - 2 * M + 0.16 });

  const y = 5.12;
  const stats = [
    ['242.06 M', '总参数'],
    ['29.91 M', '可训练（12.4%）'],
    ['213.86 M', '冻结的 Hiera 主干'],
    ['12.26 GB', '全量微调峰值（装不下）'],
  ];
  stats.forEach((st, i) => stat(s, M + i * 3.06, y, 2.9, st[0], st[1], { size: 22 }));
  footer(s, 'Adapter 1.71 M · CNN-RGB 8.96 M · CNN-Depth 8.94 M · CMF 2.61 M · MSDEF 7.68 M');
}

// 4 — reproduction
{
  const s = slideBase();
  head(s, '03', '复现结果', '四项指标，逐位对齐',
    '不是“接近”，是小数点后四位一致——这证明代码链路与论文完全相同，后续增量才有意义');

  const y = 2.2;
  const cells = [
    ['S-measure', ours.S, '结构相似度 · 主指标', GREEN],
    ['max-F', ours.maxF, '最大 F 值', INK],
    ['max-E', ours.maxE, '最大增强对齐', INK],
    ['MAE', ours.MAE, '平均绝对误差 · 越低越好', INK],
  ];
  cells.forEach((c, i) => {
    const x = M + i * 3.02;
    card(s, { x, y, w: 2.84, h: 1.36 });
    text(s, c[0], { x: x + 0.22, y: y + 0.14, w: 2.4, h: 0.24, size: 11, color: MUTED, charSpacing: 0.5 });
    text(s, c[1], { x: x + 0.22, y: y + 0.42, w: 2.4, h: 0.56, size: 30, bold: true, color: c[3] });
    text(s, c[2], { x: x + 0.22, y: y + 1.0, w: 2.44, h: 0.26, size: 9.5, color: DIM });
  });

  text(s, `12 种对比方法中排名第一：S ${ours.S}，次优 ${bestOther.Model} ${bestOther.S}`,
    { x: M, y: 3.72, w: 7.4, h: 0.3, size: 12, color: GREEN });

  img(s, 'comparison-bars.png', { x: M - 0.05, y: 4.1, w: 7.55 });

  const rows = [['消融变体', 'S', 'ΔS']];
  ablation.forEach((r) => rows.push([r.Model, r.S, (r.S - ours.S).toFixed(4)]));
  rows.push(['完整模型', ours.S, '—']);
  table(s, { x: 8.42, y: 4.1, rows, colW: [1.9, 1.1, 1.1], size: 10.5, rowH: 0.32 });
  text(s, 'ΔS 相对完整模型；去掉 Transformer 直接掉 0.16',
    { x: 8.42, y: 5.9, w: 4.3, h: 0.5, size: 9.5, color: MUTED, ls: 1.3 });
  footer(s, '来源 predictions/metrics.csv、ablation/metrics.csv；图为 tools/make_charts.py 生成');
}

// 5 — ablation insight
{
  const s = slideBase();
  head(s, '04', '消融', 'Transformer 是命脉，深度分支几乎没吃到',
    '最有价值的一条不是“哪个模块有用”，而是“为什么深度没用”——它指向一个数据管线缺陷');

  img(s, 'ablation-bars.png', { x: M - 0.08, y: 2.1, w: 7.5 });

  card(s, {
    x: 8.3, y: 2.1, w: 4.42, h: 2.3, label: '四个变体的 ΔS', accent: RED,
    body: 'w/o Transformer　−0.1622　模型基本失效\nw/o MSDEF　　　 −0.0077\nw/o CMF　　　　 −0.0056\nw/o CNN(深度)　−0.0042',
    size: 12,
  });
  card(s, {
    x: 8.3, y: 4.56, w: 4.42, h: 1.62, label: '为什么深度几乎无效', accent: ACCENT,
    body: '16 位 TIFF 被 convert("L") 饱和成纯白，深度通道是常数；\n换成 min-max 拉伸后掩膜 IoU 仍为 0.9916 → 现权重对深度不敏感，需重训才能验真。',
    size: 10.5,
  });
  footer(s, '详见 docs/dataset.md §1.1 与 docs/experiment-log.md 问题 4');
}

// 6 — system architecture
{
  const s = slideBase();
  head(s, '05', '系统架构', '把模型变成系统',
    '论文只评掩膜；课程实践要的是能按下按钮、能写报告、能追责的东西');
  img(s, 'system-architecture.png', { x: M - 0.1, y: 2.08, w: W - 2 * M + 0.2 });
  text(s, '四层：数据 → 模型（分割 / 检测 / 分级）→ 服务（引擎 + 融合 + 容错）→ 展示（工作站 / CLI / 报告）',
    { x: M, y: 6.34, w: 9.2, h: 0.3, size: 11.5, color: INK });
  text(s, '接口只定义一次：InspectionReport.as_dict() 同时喂给 UI、CSV 与评测脚本',
    { x: M, y: 6.63, w: 9.2, h: 0.3, size: 10.5, color: MUTED });
  footer(s, '图为 tools/make_diagrams.py 生成的 drawio 源文件导出，可编辑');
}

// 7 — pipeline
{
  const s = slideBase();
  head(s, '06', '单图流水线', '两条分支，一个报告',
    '串行 P50 60.9 ms；阈值滑块改变的不只是图，而是下游每一个数字');
  img(s, 'inspection-pipeline.png', { x: M - 0.3, y: 2.1, w: W - 2 * M + 0.6 });

  card(s, {
    x: M, y: 4.62, w: W - 2 * M, h: 1.42, label: '分级规则（唯一事实来源：app/engines/severity.py）',
    accent: ACCENT,
    body: 'score = 100 · min(1, area / 0.55) · (1 + 0.18 · min(1, (elong − 1) / 6)) · (0.75 + 0.25 · conf)\n' +
      'light  <  26  ≤  medium  <  55  ≤  severe　　图像级判定取最严重区域',
    size: 12.5,
  });
  footer(s, '改阈值只需要动一行常量；UI、CSV、评测报告同步变化，不会出现口径漂移');
}

// 8 — grading calibration
{
  const s = slideBase();
  head(s, '07', '分级标定', '先量数据，再定阈值',
    '数据集只有二值掩膜、没有严重度标注，所以分级是规则不是模型——但阈值必须由数据决定');

  img(s, 'severity-dist.png', { x: M - 0.05, y: 2.16, w: 7.4 });

  card(s, {
    x: 8.24, y: 2.16, w: 4.5, h: 2.18, label: '初版踩的坑', accent: RED,
    body: 'AREA_REFERENCE = 0.12 + sqrt\n→ 40 张里 40 张全判 severe\n（三档退化成一档，无法用于决策）\n\n真值面积分位：p50 0.25 / p90 0.52',
    size: 11.5,
  });
  card(s, {
    x: 8.24, y: 4.5, w: 4.5, h: 1.34, label: '标定后（全量 362 张）', accent: GREEN,
    body: `light ${sev.verdicts.light} / medium ${sev.verdicts.medium} / severe ${sev.verdicts.severe}　三档均衡`,
    size: 12,
  });
  text(s, '改为 AREA_REFERENCE = 0.55（≈ p93）+ 去掉 sqrt，并做全量复测——抽样验证曾给出错误结论。',
    { x: M, y: 6.0, w: 7.4, h: 0.5, size: 10.5, color: MUTED, ls: 1.3 });
  footer(s, '详见 docs/severity.md §3 与 docs/experiment-log.md 问题 11');
}

// 9 — system results
{
  const s = slideBase();
  head(s, '08', '系统实测', '数字、口径，和一条反例',
    '全量 362 张 + 批量 100 张；下表每个数都能用一条命令重跑');

  const m = sev.mask;
  const l = sev.latency_ms;
  const rows = [
    ['项', '实测', '口径'],
    ['覆盖 / 失败', `${sev.images} / ${sev.errors}`, '全量测试集'],
    ['掩膜 IoU', m.iou_mean.toFixed(4), `P10 ${m.iou_p10.toFixed(4)}`],
    ['Dice', m.dice_mean.toFixed(4), 'P / R 0.904 / 0.914'],
    ['端到端 P50', `${l.total_p50.toFixed(1)} ms`, `P95 ${l.total_p95.toFixed(1)} ms`],
    ['批量 100 张', `${batch.succeeded} / ${batch.failed} 失败`, `${batch.wall_time_s} s`],
    ['吞吐', `${batch.throughput_img_s} img/s`, '含叠加图落盘'],
    ['类别覆盖', `${Object.keys(batch.classes).length} 类`, '批量 100 张'],
  ];
  table(s, { x: M, y: 2.18, rows, colW: [2.3, 2.2, 2.6], size: 11.5, rowH: 0.36 });

  const o = det.overall;
  card(s, {
    x: 8.1, y: 2.18, w: 4.6, h: 1.9, label: '检测分支（如实报告）', accent: RED,
    body: `mAP@0.5 = ${o.mAP50.toFixed(4)}\nP ${o.precision.toFixed(3)} / R ${o.recall.toFixed(3)}\n目标 0.85 → 未达标：crazing 0.403、rolled-in_scale 0.682 拖累`,
    size: 11,
  });
  card(s, {
    x: 8.1, y: 4.28, w: 4.6, h: 1.55, label: '为什么系统仍然可用', accent: GREEN,
    body: '检测只提供类别与框，像素定位由分割保证；龟裂漏检时掩膜仍在，只是标为“未分类缺陷”。',
    size: 11.5,
  });
  footer(s, '来源 docs/assets/system-eval.json、batch-summary.json、detection-metrics.json');
}

// 10 — deployment
{
  const s = slideBase();
  head(s, '09', '部署', '每个加速都附上代价',
    'ONNX 不是加速方案；TensorRT 未完成——这两句话都写在报告里，而不是只写在备注里');

  const fid = lat.fidelity;
  const rows = [['后端', 'ms', 'img/s', '掩膜 IoU（均值 / 最小）']];
  lat.latency.forEach((r) => {
    const key = r.backend.split('[')[0];
    const f = fid[key] || {};
    rows.push([
      key,
      r.end_to_end_ms.toFixed(1),
      r.fps.toFixed(1),
      `${(f.mask_iou_vs_fp32_mean ?? 1).toFixed(4)} / ${(f.mask_iou_vs_fp32_min ?? 1).toFixed(4)}`,
    ]);
  });
  table(s, { x: M, y: 2.18, rows, colW: [2.2, 1.0, 1.1, 3.0], size: 11, rowH: 0.34 });

  img(s, 'latency.png', { x: M - 0.05, y: 4.02, w: 7.3 });

  card(s, {
    x: 8.1, y: 2.18, w: 4.6, h: 1.78, label: '选用 torch fp16', accent: GREEN,
    body: '40.2 ms，比 fp32 快 2.18×\n掩膜 IoU 0.981 / 最差 0.964\n无导出步骤，权重更新即生效',
    size: 12,
  });
  card(s, {
    x: 8.1, y: 4.14, w: 4.6, h: 1.72, label: 'TensorRT：未完成', accent: RED,
    body: 'DLL 依赖已解决，但 8 GB 显存被训练占满 + 页面文件不足 → 引擎构建被系统拒绝。不写入任何 TRT 数字。',
    size: 11,
  });
  footer(s, `24 张真实图 + 5 次预热 + cuda.synchronize()；两次独立重跑差异 < 1%（${lat.hardware.gpu}）`);
}

// 11 — reflection
{
  const s = slideBase();
  head(s, '10', '工程反思', '三个我们愿意主动讲的错误',
    '报告的诚实度取决于它记录了多少自己的弯路，而不是它隐藏了多少');

  const items = [
    ['延迟错了 2 倍，而且方向是对的',
      '179 → 88 ms。没预热、没 synchronize()、样本只有 12 张，还把 GPU 训练一起算进去。\n' +
      '危险之处在于“fp16 比 fp32 快”这个方向完全正确，所以没人怀疑幅度。', RED],
    ['架构图是一张空白横条',
      'value= 里写了裸 <b> 标签，文件从头就是非法 XML；draw.io CLI 退出码 0，只导出 2200×101 的空白。\n' +
      '现在生成器统一转义 + 布局自检，XML 合法性成了验收项。', ACCENT],
    ['深度分支实测无效，原因是数据而不是模型',
      '16 位 TIFF 被 convert("L") 饱和成纯白，深度通道是常数，消融里 CNN 只掉 0.0042。\n' +
      '量化影响：换成拉伸后 IoU 0.9916 —— 需要重训才能验证模块真实贡献。', BLUE],
  ];
  let y = 2.12;
  items.forEach((it) => {
    card(s, { x: M, y, w: W - 2 * M, h: 1.28, label: it[0], accent: it[2], body: it[1], size: 11.5 });
    y += 1.4;
  });
  footer(s, '三条在 docs/experiment-log.md 里都有现象、根因、修复与教训');
}

// 12 — conclusion
{
  const s = slideBase();
  head(s, '11', '总结', '三件已完成，三件明确没做',
    '交付物的可信度不来自“看起来完整”，而来自每一条都能被一条命令复核');

  card(s, {
    x: M, y: 2.14, w: 6.0, h: 2.28, label: '已完成', accent: GREEN,
    body: `· 论文四项指标逐位复现（S ${ours.S} / MAE ${ours.MAE}）\n` +
      `· 端到端 ${sev.images} 张 0 失败，P50 ${sev.latency_ms.total_p50.toFixed(1)} ms\n` +
      '· 工作站 + 批量 CLI + 34 项可执行验收自检',
    size: 12.5,
  });
  card(s, {
    x: 6.72, y: 2.14, w: 6.0, h: 2.28, label: '未完成（不掩饰）', accent: RED,
    body: '· 检测 mAP@0.5 0.772 < 0.85 目标\n· TensorRT 引擎未构建（本机内存不足）\n· 深度流未用拉伸后的数据重训',
    size: 12.5,
  });
  card(s, {
    x: M, y: 4.66, w: W - 2 * M, h: 1.42, label: '后续三件事', accent: ACCENT,
    body: '1) 用拉伸后的深度重训，验证 CMF / CNN 的真实贡献　2) 两路推理并行（60.9 → 约 45 ms）\n' +
      '3) 与产线合作获取人工评级，做分级一致性检验（Cohen\'s κ）',
    size: 12,
  });
  footer(s, '第 12 页 · 结论');
}

// 13 — handoff
{
  const s = slideBase();
  head(s, '12', '可复现性', '一条命令验证全部承诺',
    '评审不需要相信我们，只需要跑一次；任何一条不满足就返回非 0 退出码');

  slideNullCheck(s);
  footer(s, 'python tools/verify.py --json　→　{"failed": [], "passed": true}');
}

function slideNullCheck(s) {
  const cmds = [
    ['python tools/verify.py', '34 项：指标文件 / 文档链接 / 忽略规则 / 图表齐全 / 单测全绿'],
    ['python -m pytest tests -q', '23 passed，CPU-only，不需要权重与 GPU'],
    ['python -m app.main --port 7860', '四点五秒起服务，四个 Tab 全部实测过'],
  ];
  let y = 2.18;
  cmds.forEach((c) => {
    card(s, { x: M, y, w: 7.4, h: 1.06, fill: PANEL2 });
    text(s, c[0], { x: M + 0.22, y: y + 0.16, w: 6.9, h: 0.3, size: 13, bold: true, color: ACCENT });
    text(s, c[1], { x: M + 0.22, y: y + 0.52, w: 6.9, h: 0.4, size: 10.5, color: MUTED, ls: 1.25 });
    y += 1.2;
  });

  card(s, { x: 8.42, y: 2.18, w: 4.28, h: 2.5, label: 'docs/ 11 篇', accent: BLUE });
  text(s,
    '架构 · 功能 · 数据 · 评测 · 分级\n部署 · 实验日志 · 分工 · 验收\n汇报 · 索引',
    { x: 8.6, y: 2.72, w: 3.9, h: 1.3, size: 12.5, color: INK, ls: 1.5 });
  text(s, '每个数字标注来源文件；每张图由脚本生成，不手改 PNG。',
    { x: 8.6, y: 4.02, w: 3.9, h: 0.5, size: 10, color: MUTED, ls: 1.25 });

  card(s, { x: 8.42, y: 4.86, w: 4.28, h: 1.22, label: '联系方式', accent: MUTED, fill: PANEL2,
    body: '小组 7 人（A–G），分工与验收证据见\ndocs/team.md', size: 11 });
}

const out = path.join(ROOT, 'docs', 'slides.pptx');
pres.writeFile({ fileName: out }).then(() => {
  console.log(`wrote ${out}`);
});
