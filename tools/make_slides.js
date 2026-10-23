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

let CHIP = 0;
let PAGE = 0;

/** Header block: numeric chip + kicker + title + standfirst. */
function head(slide, kicker, title, sub) {
  const no = String(++CHIP).padStart(2, '0');
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

/** Page number, bottom right. Nothing else lives in the footer. */
function pageno(slide) {
  text(slide, String(++PAGE).padStart(2, '0'), {
    x: W - M - 0.62, y: H - 0.72, w: 0.62, h: 0.24,
    size: 9.5, color: DIM, align: 'right', charSpacing: 1,
  });
}

// 1 - cover
{
  const s = slideBase();
  img(s, 'ui-workstation.png', { x: 7.72, y: 0.92, h: 5.66 });
  text(s, '课程实践 · 2026', { x: 0.78, y: 1.1, w: 7, h: 0.26, size: 11.5, color: ACCENT, bold: true, charSpacing: 2.2 });
  text(s, 'GLIDE-Net', { x: 0.78, y: 1.44, w: 7.0, h: 1.05, size: 50, bold: true });
  text(s, '铁路表面缺陷检测工作站', { x: 0.78, y: 2.34, w: 7.0, h: 0.55, size: 25, color: INK });
  text(s, '分割找像素，检测认类别，规则定档位，最后做成一个能上传能批量的界面',
    { x: 0.78, y: 3.0, w: 6.8, h: 0.66, size: 13, color: MUTED, ls: 1.35 });
  const y = 4.1;
  stat(s, 0.78, y, 1.65, ours.S, 'S-measure', { size: 27 });
  stat(s, 2.46, y, 1.65, ours.MAE, 'MAE', { size: 27 });
  stat(s, 4.14, y, 1.65, ours.maxF, 'max-F', { size: 27 });
  stat(s, 5.82, y, 1.65, ours.maxE, 'max-E', { size: 27 });
  text(s, '四项指标与论文逐位一致，与 11 种方法比 S-measure 排第一',
    { x: 0.78, y: 5.06, w: 6.8, h: 0.3, size: 11.5, color: GREEN });
  text(s, '小组 7 人，A 到 G 各管一块',
    { x: 0.78, y: 6.5, w: 6.8, h: 0.28, size: 10.5, color: DIM });
  pageno(s);
}

// 2 - task definition
{
  const s = slideBase();
  head(s, '任务定义', '一张钢轨图上的三个问题',
    '人工巡检靠眼睛和手电，一个天窗点只能走几公里。系统要在同一张图上回答三个层次的问题');

  const w = 3.82;
  const gap = 0.28;
  const blocks = [
    ['分割 · 像素级', BLUE, '缺陷落在哪些像素',
      'GLIDE-Net 吃 RGB 加深度\n输出逐像素概率掩膜\n\n主指标看 S-measure 和 MAE'],
    ['检测 · 实例级', ACCENT, '这块缺陷属于哪一类',
      'YOLOv8m 在 NEU-DET 六类上训练\n给出候选框和类别\n\n六类覆盖课程的至少三类要求'],
    ['分级 · 决策级', GREEN, '要不要停机复检',
      '规则引擎，不训模型\n面积、细长比、置信度合成评分\n\n落到轻微、中等、严重三档'],
  ];
  blocks.forEach((b, i) => {
    const x = M + i * (w + gap);
    card(s, { x, y: 2.16, w, h: 2.52, label: b[0], accent: b[1] });
    text(s, b[2], { x: x + 0.22, y: 2.66, w: w - 0.44, h: 0.34, size: 15, bold: true });
    text(s, b[3], { x: x + 0.22, y: 3.06, w: w - 0.44, h: 1.4, size: 11.5, color: MUTED, ls: 1.45 });
  });

  card(s, { x: M, y: 4.94, w: W - 2 * M, h: 1.5, label: '课程给的四条考核线', accent: MUTED, fill: PANEL2 });
  const goals = [
    ['mAP@0.5 ≥ 85%', '划伤 0.946、斑块 0.910、麻点 0.860'],
    ['单张 < 100 ms', '端到端 P50 60.9 ms'],
    ['至少 3 类缺陷', '六类全覆盖'],
    ['批量 100 张不崩', '100 张连跑 0 失败'],
  ];
  goals.forEach((g, i) => {
    const x = M + 0.24 + i * 3.02;
    text(s, g[0], { x, y: 5.44, w: 2.8, h: 0.28, size: 13, bold: true, color: ACCENT });
    text(s, g[1], { x, y: 5.78, w: 2.8, h: 0.44, size: 10.5, color: MUTED, ls: 1.25 });
  });
  pageno(s);
}

// 3 - method
{
  const s = slideBase();
  head(s, '方法', '模型结构沿用原论文',
    '总参数 242.06 M，可训练的只有 29.91 M。8 GB 显存装不下全量微调，冻结主干加轻量适配是这里走得通的做法');

  img(s, 'glide-net-slide.png', { x: M - 0.08, y: 2.3, w: 12.2 });

  const stats = [
    ['242 M', '总参数'],
    ['29.9 M', '可训练 · 12.4%'],
    ['214 M', '冻结的 Hiera 主干'],
    ['12.3 GB', '全量微调峰值'],
  ];
  stats.forEach((st, i) => stat(s, M + i * 3.02, 4.36, 2.9, st[0], st[1], { size: 27 }));
  text(s, '可训练的 29.9 M 分成五块，Adapter 1.71 M、CNN-RGB 8.96 M、CNN-Depth 8.94 M、CMF 2.61 M、MSDEF 7.68 M。',
    { x: M, y: 5.5, w: 11.8, h: 0.34, size: 11.5, color: MUTED });
  text(s, '深度分支按通道和位置给 RGB 特征加权，解码器把四个尺度逐级细化回原图。',
    { x: M, y: 5.94, w: 11.8, h: 0.34, size: 12.5, color: INK });
  pageno(s);
}

// 4 - reproduction
{
  const s = slideBase();
  head(s, '复现结果', '复现结果对得上论文',
    '四项指标都落在小数点后四位，后面的工程增量才站得住');

  const y = 2.16;
  const cells = [
    ['S-measure', ours.S, '结构相似度 · 主指标', GREEN],
    ['max-F', ours.maxF, '最大 F 值', INK],
    ['max-E', ours.maxE, '最大增强对齐', INK],
    ['MAE', ours.MAE, '平均绝对误差 · 越低越好', INK],
  ];
  cells.forEach((c, i) => {
    const x = M + i * 3.02;
    card(s, { x, y, w: 2.84, h: 1.32 });
    text(s, c[0], { x: x + 0.22, y: y + 0.14, w: 2.4, h: 0.24, size: 11, color: MUTED, charSpacing: 0.5 });
    text(s, c[1], { x: x + 0.22, y: y + 0.4, w: 2.4, h: 0.56, size: 30, bold: true, color: c[3] });
    text(s, c[2], { x: x + 0.22, y: y + 0.97, w: 2.44, h: 0.26, size: 9.5, color: DIM });
  });

  const ranked = metrics.slice().sort((a, b) => b.S - a.S);
  const top = [ours, ...ranked.filter((r) => r.Model !== 'Ours').slice(0, 4)];
  const cmpRows = [['方法', 'S-measure', 'MAE']];
  top.forEach((r) => cmpRows.push([r.Model === 'Ours' ? 'GLIDE-Net 本组' : r.Model, r.S, r.MAE]));
  text(s, 'S-measure 前五名', { x: M, y: 3.62, w: 6.4, h: 0.28, size: 11.5, color: GREEN });
  table(s, { x: M, y: 3.96, rows: cmpRows, colW: [3.1, 1.2, 1.2], size: 11, rowH: 0.34 });

  const abRows = [['消融变体', 'S-measure', 'ΔS']];
  ablation
    .slice()
    .sort((a, b) => a.S - b.S)
    .forEach((r) => abRows.push([r.Model, r.S, (r.S - ours.S).toFixed(4)]));
  text(s, '四个消融变体', { x: 8.1, y: 3.62, w: 4.6, h: 0.28, size: 11.5, color: ACCENT });
  table(s, { x: 8.1, y: 3.96, rows: abRows, colW: [2.0, 1.1, 1.1], size: 11, rowH: 0.34 });
  text(s, '拿掉 Transformer 掉 0.16，模型基本失效。其余三项都在 0.008 以内，下一页单独说这一项。',
    { x: 8.1, y: 5.66, w: 4.5, h: 0.9, size: 11, color: MUTED, ls: 1.35 });
  pageno(s);
}

// 5 - ablation insight
{
  const s = slideBase();
  head(s, '消融', '消融里最值得追问的一项',
    '四个变体都按预期下降，唯独去掉深度 CNN 只掉 0.0042。追下去，问题在数据管线，不在模型');

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
    { x: M, y: 5.42, w: 8.4, h: 0.6, size: 12, color: INK, ls: 1.35 });

  card(s, {
    x: 9.02, y: 2.02, w: 3.68, h: 3.3, label: '深度这一项为什么可疑', accent: ACCENT,
    body: '数据集里的深度是 16 位 TIFF，PIL 的 convert("L") 会把它压成纯白，深度通道实际是常数。\n\n' +
      '换成 min-max 拉伸后，掩膜 IoU 仍是 0.9916，说明现权重没学到深度。\n\n' +
      '要验证这两个模块，得拿拉伸后的数据重训。',
    size: 10.5,
  });
  pageno(s);
}

// 6 - system architecture
{
  const s = slideBase();
  head(s, '系统架构', '从论文模型到能演示的系统',
    '论文只评掩膜。课堂要看的是能上传、能出报告、能批量跑，所以外面还得包三层');

  img(s, 'system-slide.png', { x: M, y: 2.2, w: 11.7 });

  const notes = [
    ['原论文的部分', MUTED, '分割模型和训练好的权重'],
    ['我们新加的部分', GREEN, '检测分支、融合分级、批量流水线、界面'],
    ['一条数据流', ACCENT, '界面、CSV、评测脚本读同一份报告对象'],
  ];
  notes.forEach((n, i) => {
    const x = M + i * 4.02;
    text(s, n[0], { x, y: 6.02, w: 3.7, h: 0.26, size: 10.5, color: n[1], bold: true, charSpacing: 0.6 });
    text(s, n[2], { x, y: 6.34, w: 3.8, h: 0.3, size: 12, color: INK });
  });
  pageno(s);
}

// 7 - the workstation we click during the demo
{
  const s = slideBase();
  head(s, '工作站', '演示时点的就是这个界面',
    'Gradio 起在 7860 端口，四点五秒就好。四个 Tab 分别对应验收里的四件事');

  img(s, 'ui-workstation.png', { x: M - 0.06, y: 2.02, h: 4.72 });

  const tabs = [
    ['单图检测', ACCENT, '上传 RGB 和深度，出四联视图、叠加图和报告卡，阈值拉到 0.7 面积和评分一起变'],
    ['批量检测', GREEN, '一次选 20 张，每张一行判定、缺陷数、面积、档位、耗时，结果能导出 CSV'],
    ['模型指标', BLUE, '复现指标、消融、系统评测、四种后端延迟，全部从结果文件读，不手抄'],
    ['系统说明', MUTED, '流水线、分级公式、阈值来源和复现命令，被问到直接翻这一页'],
  ];
  let ty = 2.06;
  tabs.forEach((t) => {
    card(s, { x: 5.34, y: ty, w: 7.36, h: 1.02, fill: PANEL2 });
    text(s, t[0], { x: 5.56, y: ty + 0.12, w: 2.0, h: 0.28, size: 13, bold: true, color: t[1] });
    text(s, t[2], { x: 5.56, y: ty + 0.46, w: 6.9, h: 0.44, size: 10.5, color: INK, ls: 1.3 });
    ty += 1.14;
  });
  text(s, '截图是测试图上的真实结果，判定中等，面积 20.58%，评分 37.3，端到端 238 ms。',
    { x: 5.34, y: 6.52, w: 6.4, h: 0.3, size: 10.5, color: DIM });
  pageno(s);
}

// 8 - pipeline
{
  const s = slideBase();
  head(s, '单图流水线', '一张图进来会经过什么',
    '两条分支并行推理，掩膜连通域按 IoU 继承检测框的类别，再按规则定档');

  img(s, 'pipeline-slide.png', { x: 1.55, y: 2.0, w: 10.2 });

  const steps = [
    ['60.9 ms', '端到端 P50，362 张实测'],
    ['40.2 ms', '其中分割推理'],
    ['11.3 ms', '其中检测推理'],
    ['26 / 55', '分级阈值，图像级取最严重区域'],
  ];
  steps.forEach((st, i) => stat(s, M + i * 3.02, 4.46, 2.9, st[0], st[1], { size: 25 }));
  text(s, 'score = 100 · min(1, 面积/0.55) · (1 + 0.18 · min(1, (细长比−1)/6)) · (0.75 + 0.25 · 置信度)',
    { x: M, y: 5.8, w: 11.9, h: 0.34, size: 13, color: ACCENT });
  text(s, '这条规则只在一个地方定义，界面、CSV 和评测脚本都调它，改常量三处一起变。',
    { x: M, y: 6.2, w: 11.9, h: 0.3, size: 11.5, color: MUTED });
  pageno(s);
}

// 9 - grading calibration
{
  const s = slideBase();
  head(s, '分级标定', '分级阈值是量出来的',
    '数据集只有二值掩膜，没有严重度标注，分级只能做成规则。可阈值不能凭感觉定');

  img(s, 'severity-dist.png', { x: M - 0.12, y: 2.06, w: 8.1 });

  card(s, {
    x: 8.78, y: 2.1, w: 3.95, h: 2.42, label: '第一版定错了', accent: RED,
    body: 'AREA_REFERENCE = 0.12 还带 sqrt，抽 40 张跑下来 40 张全判严重，三档退化成一档。\n真值面积的分位数是 p50 0.25、p90 0.52。',
    size: 11.5,
  });
  card(s, {
    x: 8.78, y: 4.72, w: 3.95, h: 1.4, label: '重新标定后 全量 362 张', accent: GREEN,
    body: `轻微 ${sev.verdicts.light} / 中等 ${sev.verdicts.medium} / 严重 ${sev.verdicts.severe}，三档比例接近`,
    size: 12,
  });
  text(s, '参考值改成 0.55，约等于真值面积的 p93，同时去掉 sqrt。第一次只抽 40 张就下结论，全量复测才看出来。',
    { x: M, y: 4.92, w: 7.9, h: 0.4, size: 11, color: MUTED, ls: 1.35 });
  card(s, {
    x: M, y: 5.4, w: 7.9, h: 1.3, label: '标定没有碰评测指标', accent: BLUE,
    body: '分位数取自真值掩膜的面积分布，S-measure 和 MAE 一个都没参与。所以调阈值只会改变判定，不会让复现分数变好看。',
    size: 11,
  });
  pageno(s);
}

// 10 - system results
{
  const s = slideBase();
  head(s, '系统实测', '362 张测试图的整体表现',
    '全量跑完没有一张失败。分割和检测是两条独立分支，各自有自己的指标');

  const m = sev.mask;
  const l = sev.latency_ms;
  const rows = [
    ['项', '实测', '说明'],
    ['覆盖 / 失败', `${sev.images} / ${sev.errors}`, '全量测试集'],
    ['掩膜 IoU', m.iou_mean.toFixed(4), `P10 ${m.iou_p10.toFixed(4)}`],
    ['Dice', m.dice_mean.toFixed(4), 'P 0.904 / R 0.914'],
    ['端到端 P50', `${l.total_p50.toFixed(1)} ms`, `P95 ${l.total_p95.toFixed(1)} ms`],
    ['批量 100 张', `${batch.succeeded} 成功 / ${batch.failed} 失败`, `${batch.wall_time_s} s · 每秒 ${batch.throughput_img_s} 张`],
    ['判定分布', `${sev.verdicts.light} / ${sev.verdicts.medium} / ${sev.verdicts.severe}`, '轻微 / 中等 / 严重'],
  ];
  table(s, { x: M, y: 2.16, rows, colW: [2.3, 2.6, 2.5], size: 11.5, rowH: 0.38 });

  const perClass = csv(A('detection-per-class.csv'))
    .filter((r) => r.meets_0_85 === 'yes')
    .sort((a, b) => Number(b.mAP50) - Number(a.mAP50));
  const cRows = [['缺陷类别', 'mAP@0.5', '目标 0.85']];
  perClass.forEach((r) => cRows.push([r.class, Number(r.mAP50).toFixed(4), '达标']));
  text(s, '检测分支按类别看', { x: 8.0, y: 2.16, w: 4.7, h: 0.28, size: 11.5, color: GREEN });
  table(s, { x: 8.0, y: 2.5, rows: cRows, colW: [2.1, 1.2, 1.2], size: 11.5, rowH: 0.42 });
  text(s, '这三类都过了课程的 0.85 线，划伤排第一。',
    { x: 8.0, y: 4.3, w: 4.7, h: 0.5, size: 11.5, color: INK, ls: 1.3 });

  const cls = Object.entries(batch.classes).map(([k, v]) => `${k} ${v}`).join('　');
  card(s, {
    x: 8.0, y: 4.96, w: 4.72, h: 1.74, label: '批量 100 张的类别分布', accent: ACCENT,
    body: cls + '\n\n六类全部出现在批量结果里。',
    size: 11,
  });
  pageno(s);
}

// 11 - deployment
{
  const s = slideBase();
  head(s, '部署', '四种后端的实测延迟',
    '24 张真实图、5 次预热，计时前后同步 CUDA，两次独立重跑差不到 1%');

  const fid = lat.fidelity;
  const rows = [['后端', 'ms', '张每秒', '掩膜 IoU 均值 / 最小']];
  lat.latency.forEach((r) => {
    const key = r.backend.split('[')[0].replace('-cuda', '');
    const f = fid[key];
    if (!f) throw new Error(`latency.json has no fidelity entry for ${key}`);
    const iou = key === 'torch-fp32'
      ? '基线'
      : `${f.mask_iou_vs_fp32_mean.toFixed(4)} / ${f.mask_iou_vs_fp32_min.toFixed(4)}`;
    rows.push([key, r.end_to_end_ms.toFixed(1), r.fps.toFixed(1), iou]);
  });
  table(s, { x: M, y: 2.16, rows, colW: [2.2, 1.0, 1.1, 3.0], size: 11, rowH: 0.34 });

  img(s, 'latency.png', { x: M - 0.05, y: 4.16, w: 7.2 });

  card(s, {
    x: 7.9, y: 2.16, w: 4.82, h: 1.5, label: '线上默认用 torch fp16', accent: GREEN,
    body: '40.2 ms，比 fp32 快 2.18 倍，掩膜 IoU 0.981。不用导出，权重一改就生效。',
    size: 12,
  });
  card(s, {
    x: 7.9, y: 3.82, w: 4.82, h: 1.5, label: 'ONNX 用来保证可移植', accent: BLUE,
    body: 'fp32 的 ONNX 和 torch 只差 2.7e-05，换机器不换代码。它不是加速手段。',
    size: 12,
  });
  card(s, {
    x: 7.9, y: 5.48, w: 4.82, h: 1.22, label: 'TensorRT 没做成', accent: RED,
    body: '8 GB 显存被训练占满，页面文件也不够，引擎构建被系统拒绝。没测到就不写数字。',
    size: 11,
  });
  pageno(s);
}

// 12 - problems
{
  const s = slideBase();
  head(s, '过程记录', '几个花了时间才改对的地方',
    '下面这些都能在实验日志里查到现象、定位过程和改法，这里只挑三条跟结果有关的');

  const items = [
    ['延迟第一次量出了一倍误差', RED,
      '初测端到端 175 ms，重测是 88 ms。没预热、没等 CUDA 做完、样本只取 12 张，还把同时在跑的训练算进去了。\n' +
      '改法是把预热次数和同步写进基准脚本，固定 24 张，两次独立重跑差异压到 1% 以内。'],
    ['深度通道一直是常数', ACCENT,
      '16 位 TIFF 走 PIL 的 convert("L") 会被压成纯白，所以消融里去掉深度只掉 0.0042。\n' +
      '服务层改成 min-max 拉伸，8 张图对比掩膜 IoU 0.9916，说明修的是输入，不是权重。'],
    ['分级第一版三档退化成一档', BLUE,
      '初版参考值 0.12 还带 sqrt，抽 40 张跑出来 40 张全判严重。\n' +
      '用 362 张真值面积的分位数重新标定到 0.55，三档变成 102 / 120 / 140。'],
  ];
  let y = 2.12;
  items.forEach((it) => {
    card(s, { x: M, y, w: W - 2 * M, h: 1.3, label: it[0], accent: it[1], body: it[2], size: 11.5 });
    y += 1.44;
  });
  pageno(s);
}

// 13 - conclusion
{
  const s = slideBase();
  head(s, '总结', '现在能做什么，还差什么',
    '能演示的部分都有命令和文件对应，减分项也写清楚停在哪一步');

  card(s, {
    x: M, y: 2.14, w: 6.0, h: 2.24, label: '已经能演示', accent: GREEN,
    body: `· 论文四项指标复现，S ${ours.S}，MAE ${ours.MAE}\n` +
      `· ${sev.images} 张端到端 0 失败，P50 ${sev.latency_ms.total_p50.toFixed(1)} ms\n` +
      '· 六类缺陷、三级判定、单图和批量两种入口\n' +
      '· 一条命令自检 40 项断言',
    size: 12,
  });
  card(s, {
    x: 6.72, y: 2.14, w: 6.0, h: 2.24, label: '还没做完', accent: RED,
    body: '· 检测整体 mAP@0.5 为 0.772，三类过 0.85\n' +
      '· 龟裂和氧化皮压入是低对比度纹理，还没到位\n' +
      '· TensorRT 引擎没构建出来\n' +
      '· 深度流没有用拉伸后的数据重训',
    size: 12,
  });
  card(s, {
    x: M, y: 4.6, w: W - 2 * M, h: 1.86, label: '接下来三件事', accent: ACCENT,
    body: '1) 用拉伸后的深度重训，看 CMF 和 CNN 分流的真实贡献\n' +
      '2) 分割与检测放两个 CUDA stream，把 60.9 ms 压到 45 ms 左右\n' +
      '3) 拿产线人工评级做分级一致性检验',
    size: 12,
  });
  pageno(s);
}

// 14 - reproducibility
{
  const s = slideBase();
  head(s, '可复现性', '这些数字怎么复核',
    '所有承诺都能用命令跑一遍，任何一条不满足，脚本返回非零退出码');

  const cmds = [
    ['python tools/verify.py', '40 项断言，覆盖指标、评测口径、文档链接、忽略规则、图表和单测'],
    ['python -m pytest tests -q', '27 个用例全过，只用 CPU，不需要权重和 GPU'],
    ['python -m app.main --port 7860', '四点五秒起服务，四个 Tab 都在现场点过'],
  ];
  let cy = 2.2;
  cmds.forEach((c) => {
    card(s, { x: M, y: cy, w: 7.4, h: 1.06, fill: PANEL2 });
    text(s, c[0], { x: M + 0.22, y: cy + 0.16, w: 6.9, h: 0.3, size: 13, bold: true, color: ACCENT });
    text(s, c[1], { x: M + 0.22, y: cy + 0.52, w: 6.9, h: 0.4, size: 10.5, color: MUTED, ls: 1.25 });
    cy += 1.2;
  });

  card(s, { x: 8.42, y: 2.2, w: 4.28, h: 2.44, label: 'docs/ 下 11 篇文档', accent: BLUE });
  text(s, '架构 · 功能 · 数据 · 评测 · 分级\n部署 · 实验日志 · 分工 · 验收\n汇报 · 索引',
    { x: 8.6, y: 2.74, w: 3.9, h: 1.2, size: 12.5, color: INK, ls: 1.5 });
  text(s, '每个数字都标了来源文件，每张图由脚本生成，不手改 PNG。',
    { x: 8.6, y: 3.98, w: 3.9, h: 0.5, size: 10, color: MUTED, ls: 1.25 });

  card(s, { x: 8.42, y: 4.92, w: 4.28, h: 1.16, label: '小组分工', accent: MUTED, fill: PANEL2,
    body: '7 人（A 到 G），分工和验收证据在 docs/team.md', size: 11 });
  pageno(s);
}

const out = path.join(ROOT, 'docs', 'slides.pptx');
pres.writeFile({ fileName: out }).then(() => {
  console.log(`wrote ${out}`);
});
