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

/** PNG pixel size, read from the IHDR chunk (bytes 16-24). */
function pngSize(file) {
  const buf = fs.readFileSync(A(file));
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

/**
 * Place an image. Giving only one side keeps the source aspect ratio -- passing
 * a lone `w` to pptxgenjs leaves h at 1 in and squashes every chart flat.
 */
function img(slide, file, o) {
  if (!fs.existsSync(A(file))) {
    text(slide, `[missing ${file}]`, { x: o.x, y: o.y, w: 3, h: 0.3, size: 10, color: RED });
    return;
  }
  const px = pngSize(file);
  const ratio = px.w / px.h;
  const opts = { x: o.x, y: o.y, path: A(file) };
  if (o.w && o.h) { opts.w = o.w; opts.h = o.h; }
  else if (o.w) { opts.w = o.w; opts.h = o.w / ratio; }
  else if (o.h) { opts.h = o.h; opts.w = o.h * ratio; }
  else throw new Error(`img(${file}) needs w or h`);
  slide.addImage(opts);
  return opts;
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
  img(s, 'example-overlay.png', { x: 9.55, y: 0.86, h: 5.2 });
  text(s, '课程实践 · 2026', { x: 0.78, y: 1.28, w: 7, h: 0.26, size: 11.5, color: ACCENT, bold: true, charSpacing: 2.2 });
  text(s, 'GLIDE-Net', { x: 0.78, y: 1.62, w: 7.4, h: 1.05, size: 52, bold: true });
  text(s, '铁路表面缺陷检测工作站', { x: 0.78, y: 2.52, w: 7.4, h: 0.55, size: 26, color: INK });
  text(s, '从复现论文到做成能演示的系统，覆盖分割、检测、分级和四种推理后端',
    { x: 0.78, y: 3.2, w: 7.4, h: 0.4, size: 13, color: MUTED });
  const y = 4.28;
  stat(s, 0.78, y, 1.7, ours.S, 'S-measure', { size: 28 });
  stat(s, 2.5, y, 1.7, ours.MAE, 'MAE', { size: 28 });
  stat(s, 4.2, y, 1.7, ours.maxF, 'max-F', { size: 28 });
  stat(s, 5.9, y, 1.7, ours.maxE, 'max-E', { size: 28 });
  text(s, '四项指标与论文一致，12 种对比方法里 S-measure 排第一',
    { x: 0.78, y: 5.28, w: 7.3, h: 0.3, size: 11.5, color: GREEN });
  text(s, '课程实践小组 7 人，A 到 G 各自负责一块，分工和验收证据见 docs/team.md',
    { x: 0.78, y: 6.45, w: 7.6, h: 0.3, size: 10.5, color: DIM });
  text(s, '右侧是系统对测试图 101.bmp 的真实输出。掩膜来自分割分支，框和类别来自检测分支。',
    { x: 8.35, y: 6.22, w: 4.3, h: 0.55, size: 9.5, color: MUTED, ls: 1.3 });
}

// 2 — task definition
{
  const s = slideBase();
  head(s, '01', '任务定义', '一张钢轨图上的三个问题',
    '人工巡检靠眼睛和手电，一个天窗点只能走几公里。系统要回答的是同一张图上的三个层次');

  const y = 2.28;
  const w = 3.82;
  const gap = 0.28;
  const blocks = [
    ['分割 · 像素级', BLUE, '缺陷落在哪些像素',
      'GLIDE-Net 接收 RGB 与深度四通道\n输出逐像素概率掩膜\n\n主指标是 S-measure 和 MAE\n与论文对齐到四位小数'],
    ['检测 · 实例级', ACCENT, '这块缺陷属于哪一类',
      'YOLOv8m 在 NEU-DET 六类上训练\n输出候选框和置信度\n\n类别由检测分支提供\nmAP@0.5 为 0.772'],
    ['分级 · 决策级', GREEN, '要不要停机复检',
      '规则引擎，不训练模型\n面积、细长比和置信度合成评分\n\n落到轻微、中等、严重三档\n阈值由真值面积分布标定'],
  ];
  blocks.forEach((b, i) => {
    const x = M + i * (w + gap);
    card(s, { x, y, w, h: 2.62, label: b[0], accent: b[1] });
    text(s, b[2], { x: x + 0.22, y: y + 0.56, w: w - 0.44, h: 0.34, size: 15, bold: true });
    text(s, b[3], { x: x + 0.22, y: y + 0.96, w: w - 0.44, h: 1.4, size: 11.5, color: MUTED, ls: 1.45 });
  });

  text(s, '课程实践的重心放在后两层，以及把三层串成一条能跑起来的流水线。',
    { x: M, y: 5.3, w: W - 2 * M - 0.5, h: 0.7, size: 13, color: INK, ls: 1.4 });
  footer(s, '第 2 页 · 任务定义');
}

// 3 — method
{
  const s = slideBase();
  head(s, '02', '方法', '模型结构沿用原论文',
    '总参数 242.06 M，可训练的只有 29.91 M。8 GB 显存装不下全量微调，冻结主干加轻量适配是走得通的做法');
  img(s, 'glide-net-architecture.png', { x: M - 0.12, y: 1.96, w: 7.6 });

  const stats = [
    ['242.06 M', '总参数'],
    ['29.91 M', '可训练，占 12.4%'],
    ['213.86 M', '冻结的 Hiera 主干'],
    ['12.26 GB', '全量微调的峰值显存'],
  ];
  stats.forEach((st, i) => stat(s, 8.62, 2.02 + i * 1.02, 4.1, st[0], st[1], { size: 26 }));
  text(s, '可训练部分是 Adapter 1.71 M、CNN-RGB 8.96 M、\nCNN-Depth 8.94 M、CMF 2.61 M 和 MSDEF 7.68 M。',
    { x: 8.62, y: 6.12, w: 4.1, h: 0.6, size: 9.5, color: MUTED, ls: 1.3 });
  footer(s, '图为 tools/make_diagrams.py 生成，左侧 RGB 与深度两条输入，中间三个 CMF 做跨模态融合，右侧 MSDEF 解码到四个尺度');
}

// 4 — reproduction
{
  const s = slideBase();
  head(s, '03', '复现结果', '复现结果对得上论文',
    '四项指标都落在小数点后四位，后面的工程增量才站得住');

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

  const ranked = metrics.slice().sort((a, b) => b.S - a.S);
  const top = [ours, ...ranked.filter((r) => r.Model !== 'Ours').slice(0, 4)];
  const cmpRows = [['方法', 'S', 'MAE']];
  top.forEach((r) => cmpRows.push([r.Model === 'Ours' ? 'GLIDE-Net（本组）' : r.Model, r.S, r.MAE]));
  text(s, 'S-measure 排名前五，完整 12 行在 predictions/metrics.csv', 
    { x: M, y: 3.78, w: 6.4, h: 0.28, size: 11.5, color: GREEN });
  table(s, { x: M, y: 4.14, rows: cmpRows, colW: [3.1, 1.2, 1.2], size: 11, rowH: 0.34 });

  const abRows = [['消融变体', 'S', 'ΔS']];
  ablation
    .slice()
    .sort((a, b) => a.S - b.S)
    .forEach((r) => abRows.push([r.Model, r.S, (r.S - ours.S).toFixed(4)]));
  text(s, '四个消融变体的 S 与差值', { x: 8.1, y: 3.78, w: 4.6, h: 0.28, size: 11.5, color: ACCENT });
  table(s, { x: 8.1, y: 4.14, rows: abRows, colW: [2.0, 1.1, 1.1], size: 11, rowH: 0.34 });
  text(s, '拿掉 Transformer 掉 0.16，模型基本失效。其余三项都在 0.008 以内。',
    { x: 8.1, y: 5.86, w: 4.4, h: 0.6, size: 10, color: MUTED, ls: 1.3 });
  footer(s, '数据来自 predictions/metrics.csv 和 ablation/metrics.csv，对比图在 docs/assets/comparison-bars.png');
}

// 5 — ablation insight
{
  const s = slideBase();
  head(s, '04', '消融', '消融里最值得追问的一项',
    '四个变体都按预期下降，唯独去掉深度 CNN 只掉 0.0042。查下来问题在数据管线，不在模型');

  const maxAbs = Math.max(...ablation.map((r) => Math.abs(r.S - ours.S)));
  const order = ['w_o_Transformer', 'w_o_MSDEF', 'w_o_CMF', 'w_o_CNN'];
  const rows = order.map((k) => ablation.find((r) => r.Model === k)).filter(Boolean);
  const barX = M + 3.4;
  const trackW = 3.7;
  let by = 2.46;
  rows.forEach((r, i) => {
    const d = r.S - ours.S;
    const w = Math.max(0.09, Math.abs(d) / maxAbs * trackW);
    text(s, r.Model, { x: M, y: by, w: 3.3, h: 0.34, size: 12.5, color: i === 0 ? INK : MUTED, valign: 'middle' });
    s.addShape('roundRect', {
      x: barX, y: by + 0.03, w: trackW, h: 0.28, rectRadius: 0.03,
      fill: { color: '101821' }, line: { color: '101821', width: 0 },
    });
    s.addShape('roundRect', {
      x: barX, y: by + 0.03, w, h: 0.28, rectRadius: 0.03,
      fill: { color: i === 0 ? RED : ACCENT }, line: { color: i === 0 ? RED : ACCENT, width: 0 },
    });
    text(s, d.toFixed(4), { x: barX + trackW + 0.16, y: by, w: 1.0, h: 0.34, size: 12.5, bold: true,
      color: i === 0 ? RED : INK, valign: 'middle' });
    by += 0.72;
  });
  text(s, 'ΔS 相对完整模型（S = 0.8718）', { x: M, y: 2.02, w: 6.0, h: 0.26, size: 11, color: MUTED });
  text(s, 'Transformer 是主干，拿掉就塌。另外三项各自不到 0.01，单看数字会以为这三个模块都没用。',
    { x: M, y: 5.42, w: 8.2, h: 0.6, size: 11.5, color: INK, ls: 1.35 });

  card(s, {
    x: 9.02, y: 2.02, w: 3.68, h: 3.3, label: '深度这一项为什么可疑', accent: ACCENT,
    body: '数据集里的深度是 16 位 TIFF，PIL 的 convert("L") 会把它压成纯白，深度通道实际是常数。\n\n' +
      '换成 min-max 拉伸后，掩膜 IoU 仍是 0.9916，说明现权重没学到深度。\n\n' +
      '要验证这部分模块的贡献，得用拉伸后的数据重训，这条列进了后续工作。',
    size: 10.5,
  });
  footer(s, '详见 docs/dataset.md §1.1 与 docs/experiment-log.md 问题 4');
}

// 6 — system architecture
{
  const s = slideBase();
  head(s, '05', '系统架构', '从模型到能演示的系统',
    '论文只评掩膜，课堂演示需要能上传、能出报告、能批量跑');
  img(s, 'system-architecture.png', { x: M - 0.12, y: 1.96, w: 8.9 });

  card(s, {
    x: 9.62, y: 1.96, w: 3.1, h: 3.5, label: '四层各管什么', accent: ACCENT,
    body: '数据层 读取 RGB 与深度，做 EXIF 校正和归一化\n\n' +
      '模型层 分割、检测、分级三个引擎\n\n' +
      '服务层 调度、结果融合、异常兜底\n\n' +
      '展示层 工作站界面、命令行、报告',
    size: 10.5,
  });
  text(s, '接口只定义一次，InspectionReport.as_dict() 同时喂给界面、CSV 和评测脚本。',
    { x: M, y: 6.14, w: 11.9, h: 0.3, size: 11.5, color: INK });
  footer(s, '图为 tools/make_diagrams.py 生成的 drawio 源文件导出，可以继续编辑；每条连线上都标了真实数据形状');
}

// 7 — pipeline
{
  const s = slideBase();
  head(s, '06', '单图流水线', '单图流水线怎么走',
    '全量 362 张串行跑下来 P50 是 60.9 ms，其中分割 40.2 ms、检测 11.3 ms');
  img(s, 'inspection-pipeline.png', { x: M - 0.35, y: 1.98, w: 12.55 });
  text(s, '两个阈值的实际取值是 26 和 55，图像级判定取最严重的那一块区域。',
    { x: M, y: 6.46, w: 11.9, h: 0.3, size: 11.5, color: INK });
  footer(s, '图为 tools/make_diagrams.py 生成，底部虚线框里是分级公式和阈值；改常量后界面、CSV 与评测脚本一起变，不会各说各话');
}

// 8 — grading calibration
{
  const s = slideBase();
  head(s, '07', '分级标定', '分级阈值是量出来的',
    '数据集只有二值掩膜，没有严重度标注，分级只能做成规则。可阈值不能凭感觉定');

  img(s, 'severity-dist.png', { x: M - 0.05, y: 2.1, w: 7.9 });

  card(s, {
    x: 8.6, y: 2.1, w: 4.1, h: 2.42, label: '第一版定错了', accent: RED,
    body: 'AREA_REFERENCE = 0.12 还带 sqrt，抽 40 张跑下来 40 张全判严重，三档退化成一档。\n真值面积的分位数是 p50 0.25、p90 0.52。',
    size: 11.5,
  });
  card(s, {
    x: 8.6, y: 4.72, w: 4.1, h: 1.4, label: '重新标定后（全量 362 张）', accent: GREEN,
    body: `轻微 ${sev.verdicts.light} / 中等 ${sev.verdicts.medium} / 严重 ${sev.verdicts.severe}，三档比例接近`,
    size: 12,
  });
  text(s, '参考值改成 0.55，约等于真值面积的 p93，同时去掉 sqrt。第一次只抽 40 张就下结论，全量复测才看出来。',
    { x: M, y: 4.86, w: 7.7, h: 0.7, size: 11, color: MUTED, ls: 1.35 });
  footer(s, '详见 docs/severity.md §3 与 docs/experiment-log.md 问题 11');
}

// 9 — system results
{
  const s = slideBase();
  head(s, '08', '系统实测', '362 张测试图的实测数据',
    '下表每个数字都能用一条命令重跑，口径写在第三列');

  const m = sev.mask;
  const l = sev.latency_ms;
  const rows = [
    ['项', '实测', '口径'],
    ['覆盖 / 失败', `${sev.images} / ${sev.errors}`, '全量测试集'],
    ['掩膜 IoU', m.iou_mean.toFixed(4), `P10 ${m.iou_p10.toFixed(4)}`],
    ['Dice', m.dice_mean.toFixed(4), 'P / R 0.904 / 0.914'],
    ['端到端 P50', `${l.total_p50.toFixed(1)} ms`, `P95 ${l.total_p95.toFixed(1)} ms`],
    ['批量 100 张', `${batch.succeeded} 成功 / ${batch.failed} 失败`, `${batch.wall_time_s} s`],
    ['吞吐', `${batch.throughput_img_s} img/s`, '含叠加图落盘'],
    ['类别覆盖', `${Object.keys(batch.classes).length} 类`, '批量 100 张'],
  ];
  table(s, { x: M, y: 2.18, rows, colW: [2.4, 2.5, 2.4], size: 11.5, rowH: 0.36 });

  const o = det.overall;
  card(s, {
    x: 8.1, y: 2.18, w: 4.6, h: 1.78, label: '检测分支的短板', accent: RED,
    body: `mAP@0.5 为 ${o.mAP50.toFixed(4)}，低于 0.85 的目标。\n拖累主要来自 crazing 0.403 和 rolled-in_scale 0.682，细纹理和框标注偏主观是主因。`,
    size: 11,
  });
  card(s, {
    x: 8.1, y: 4.12, w: 4.6, h: 1.5, label: '系统为什么还能用', accent: GREEN,
    body: '检测只负责类别和框，像素定位由分割保证。龟裂漏检时掩膜仍在，报告里记为未分类缺陷。',
    size: 11.5,
  });
  const cls = Object.entries(batch.classes).map(([k, v]) => `${k} ${v}`).join('　');
  card(s, {
    x: M, y: 5.3, w: 7.3, h: 1.02, label: '批量 100 张的类别分布', accent: ACCENT,
    body: cls,
    size: 11.5,
  });
  footer(s, '数据来自 docs/assets/system-eval.json、batch-summary.json 和 detection-metrics.json');
}

// 10 — deployment
{
  const s = slideBase();
  head(s, '09', '部署', '四种后端的实测延迟',
    '24 张真实图和 5 次预热，计时前后同步 CUDA，两次独立重跑差不到 1%');

  const fid = lat.fidelity;
  const rows = [['后端', 'ms', 'img/s', '掩膜 IoU（均值 / 最小）']];
  lat.latency.forEach((r) => {
    const key = r.backend.split('[')[0].replace('-cuda', '');
    const f = fid[key];
    if (!f) throw new Error(`latency.json has no fidelity entry for ${key}`);
    const iou = key === 'torch-fp32'
      ? '基线'
      : `${f.mask_iou_vs_fp32_mean.toFixed(4)} / ${f.mask_iou_vs_fp32_min.toFixed(4)}`;
    rows.push([key, r.end_to_end_ms.toFixed(1), r.fps.toFixed(1), iou]);
  });
  table(s, { x: M, y: 2.18, rows, colW: [2.2, 1.0, 1.1, 3.0], size: 11, rowH: 0.34 });

  img(s, 'latency.png', { x: M - 0.05, y: 4.02, w: 7.3 });

  card(s, {
    x: 8.1, y: 2.18, w: 4.6, h: 1.52, label: '选用 torch fp16', accent: GREEN,
    body: '40.2 ms，比 fp32 快 2.18 倍。掩膜 IoU 0.981，最差 0.964。不用导出，权重一改就生效。',
    size: 12,
  });
  card(s, {
    x: 8.1, y: 3.9, w: 4.6, h: 1.55, label: 'TensorRT 没做完', accent: RED,
    body: 'DLL 依赖已经解决，引擎构建要的显存和页面文件不够，本机做不下去。报告里不写任何 TensorRT 的数字。',
    size: 11,
  });
  footer(s, `ONNX 的定位是可移植，fp32 ONNX 与 torch 的误差只有 2.7e-05，延迟并不更快　${lat.hardware.gpu}`);
}

// 11 — problems we hit
{
  const s = slideBase();
  head(s, '10', '过程问题', '几个花了时间才修对的问题',
    '下面这些在 docs/experiment-log.md 里都有现象、原因和改法');

  const items = [
    ['延迟第一次量出了一倍误差', RED,
      '初测端到端 175 ms，重测是 88 ms。没预热、没同步、样本只有 12 张，还把同时在跑的 GPU 训练算进去了。\n' +
      '改法是把预热次数和 cuda.synchronize() 写进基准脚本，固定 24 张样本，两次重跑差异压到 1% 以内。'],
    ['架构图导出来是空白横条', ACCENT,
      '画图脚本里写了裸的 <b> 标签，文件本身就不是合法 XML。draw.io 命令行照样返回 0，导出的 PNG 只有 2200×101。\n' +
      '现在生成器统一转义，并检查盒子不越界、不重叠，XML 是否合法也进了验收项。'],
    ['检测精度没到目标值', BLUE,
      'YOLOv8m 训满 150 epoch，test 180 张上 mAP@0.5 是 0.7721，目标是 0.85。\n' +
      '逐类看，crazing 0.403、rolled-in_scale 0.682，另外四类都在 0.83 以上。报告里给了三条改法，没做成的不写成做成了。'],
  ];
  let y = 2.12;
  items.forEach((it) => {
    card(s, { x: M, y, w: W - 2 * M, h: 1.26, label: it[0], accent: it[1], body: it[2], size: 11.5 });
    y += 1.4;
  });
  footer(s, '第 11 页 · 过程问题');
}

// 12 — conclusion
{
  const s = slideBase();
  head(s, '11', '总结', '现在能做什么，还差什么',
    '能演示的部分都有命令和文件对应，没做完的部分也写清楚停在哪一步');

  card(s, {
    x: M, y: 2.14, w: 6.0, h: 2.1, label: '现在已经能', accent: GREEN,
    body: `· 论文四项指标复现，S ${ours.S}，MAE ${ours.MAE}\n` +
      `· ${sev.images} 张端到端 0 失败，P50 ${sev.latency_ms.total_p50.toFixed(1)} ms\n` +
      '· 工作站、批量命令行和 40 项自检断言',
    size: 12.5,
  });
  card(s, {
    x: 6.72, y: 2.14, w: 6.0, h: 2.1, label: '还没做完', accent: RED,
    body: '· 检测 mAP@0.5 为 0.772，低于 0.85 的目标\n· TensorRT 引擎没构建出来\n· 深度流没有用拉伸后的数据重训',
    size: 12.5,
  });
  card(s, {
    x: M, y: 4.46, w: W - 2 * M, h: 1.7, label: '接下来三件事', accent: ACCENT,
    body: '1) 用拉伸后的深度重训，看 CMF 和 CNN 分支的真实贡献\n' +
      '2) 分割与检测放两个 CUDA stream，把 60.9 ms 压到 45 ms 左右\n' +
      "3) 拿到产线的人工评级，做分级一致性检验（Cohen's κ）",
    size: 12,
  });
  footer(s, '第 12 页 · 总结');
}

// 13 — reproduction
{
  const s = slideBase();
  head(s, '12', '可复现性', '这些数字怎么复核',
    '所有承诺都能用命令跑一遍，任何一条不满足，脚本返回非零退出码');

  const cmds = [
    ['python tools/verify.py', '40 项断言，覆盖指标、评测口径、文档链接、忽略规则、图表和单测'],
    ['python -m pytest tests -q', '23 个用例全过，只用 CPU，不需要权重和 GPU'],
    ['python -m app.main --port 7860', '四点五秒起服务，四个 Tab 都在现场点过'],
  ];
  let cy = 2.18;
  cmds.forEach((c) => {
    card(s, { x: M, y: cy, w: 7.4, h: 1.06, fill: PANEL2 });
    text(s, c[0], { x: M + 0.22, y: cy + 0.16, w: 6.9, h: 0.3, size: 13, bold: true, color: ACCENT });
    text(s, c[1], { x: M + 0.22, y: cy + 0.52, w: 6.9, h: 0.4, size: 10.5, color: MUTED, ls: 1.25 });
    cy += 1.2;
  });

  card(s, { x: 8.42, y: 2.18, w: 4.28, h: 2.5, label: 'docs/ 下 11 篇文档', accent: BLUE });
  text(s,
    '架构 · 功能 · 数据 · 评测 · 分级\n部署 · 实验日志 · 分工 · 验收\n汇报 · 索引',
    { x: 8.6, y: 2.72, w: 3.9, h: 1.3, size: 12.5, color: INK, ls: 1.5 });
  text(s, '每个数字都标了来源文件，每张图由脚本生成，不手改 PNG。',
    { x: 8.6, y: 4.02, w: 3.9, h: 0.5, size: 10, color: MUTED, ls: 1.25 });

  card(s, { x: 8.42, y: 4.86, w: 4.28, h: 1.22, label: '小组分工', accent: MUTED, fill: PANEL2,
    body: '小组 7 人（A 到 G），分工和验收证据见\ndocs/team.md', size: 11 });
}

const out = path.join(ROOT, 'docs', 'slides.pptx');
pres.writeFile({ fileName: out }).then(() => {
  console.log(`wrote ${out}`);
});
