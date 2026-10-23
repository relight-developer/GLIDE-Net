/**
 * Render the three presentation diagrams from Mermaid source.
 *
 * The draw.io figures in docs/assets/*.drawio stay as the detailed reference
 * (every tensor shape, every edge). They are too dense for a projector, so the
 * slides use these: at most six boxes, one idea per row, nothing below 22 px.
 *
 * Pipeline: Mermaid text -> pele (SVG) -> headless Chrome (PNG).
 *
 *   node tools/make_mermaid.mjs            # write .mmd/.svg/.png into docs/assets
 *
 * Every rendered PNG is checked, because a valid-looking export can still be a
 * thin blank strip: the script measures the ink bounding box and refuses to
 * report success unless the drawing fills the frame.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execFileSync } from 'child_process';
import { render } from 'pele';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const OUT = path.join(ROOT, 'docs', 'assets');

const BG = '#0B0F14';
const INK = '#E8EEF4';
const DIM = '#8695A4';
const YELLOW = '#F5C542';
const GREEN = '#30A46C';
const BLUE = '#4C9AFF';
const RED = '#E5484D';
const PANEL = '#16212C';
const LINE = '#4C6A85';

// One style vocabulary for all three figures: yellow means "the model does the
// work here", green means "our code", blue means "input or output".
const CLASSES = `
  classDef io fill:${PANEL},stroke:${LINE},stroke-width:2px,color:${INK},font-size:24px
  classDef frozen fill:${PANEL},stroke:${BLUE},stroke-width:2px,color:${INK},font-size:24px
  classDef model fill:${PANEL},stroke:${YELLOW},stroke-width:3px,color:${INK},font-size:24px
  classDef ours fill:#1B2A22,stroke:${GREEN},stroke-width:3px,color:${INK},font-size:24px
  classDef out fill:#101820,stroke:${RED},stroke-width:2px,color:${INK},font-size:24px
`;

const DIAGRAMS = {
  // --- replace docs/assets/glide-net-architecture.png on the slides
  'glide-net-slide.png': `flowchart LR
  IN["输入<br/>RGB + 深度<br/>352 × 352"]:::io
  TR["冻结主干<br/>SAM2 Hiera-L<br/>213.86 M"]:::frozen
  CMF["跨模态融合<br/>深度加权<br/>三级各一次"]:::model
  DEC["MSDEF 解码<br/>四尺度<br/>细化边缘"]:::model
  OUT["缺陷掩膜<br/>352 × 352"]:::out
  IN --> TR --> CMF --> DEC --> OUT
${CLASSES}`,

  // --- replace docs/assets/system-architecture.png
  'system-slide.png': `flowchart LR
  D1["RSDDS-AUG<br/>1862 张"]:::io
  D2["NEU-DET<br/>1797 张"]:::io
  M1["分割<br/>像素掩膜"]:::model
  M2["检测<br/>类别框"]:::model
  S["服务层<br/>融合分级批量"]:::ours
  U["工作站<br/>四个页面"]:::out
  D1 --> M1 --> S
  D2 --> M2 --> S
  S --> U
${CLASSES}`,

  // --- replace docs/assets/inspection-pipeline.png
  'pipeline-slide.png': `flowchart LR
  UP["上传<br/>RGB + 深度"]:::io
  PRE["预处理<br/>352 × 352"]:::io
  SEG["分割<br/>40 ms"]:::model
  DET["检测<br/>11 ms"]:::model
  FUSE["融合与分级<br/>IoU 继承类别"]:::ours
  REP["报告<br/>JSON + 叠加图"]:::out
  UP --> PRE --> SEG --> FUSE --> REP
  PRE --> DET --> FUSE
${CLASSES}`,
};

const PAGE = (svg, w, h) => `<!doctype html><meta charset="utf-8">
<style>
  html,body{margin:0;padding:0;background:${BG};width:${w}px;height:${h}px;overflow:hidden}
  .pele{--pele-bg:${BG};--pele-fg:${INK};--pele-muted:${DIM};--pele-line:${LINE};
        --pele-surface:${PANEL};--pele-surface-alt:#1B2531;--pele-border:#2A3A48;
        font-family:'Segoe UI','Microsoft YaHei',sans-serif;display:block}
  svg{display:block;width:${w}px;height:${h}px}
</style>${svg}`;

function findChrome() {
  const cands = [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    '/usr/bin/google-chrome',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ];
  const hit = cands.find((p) => fs.existsSync(p));
  if (!hit) throw new Error('no Chrome/Edge binary found for SVG rasterising');
  return hit;
}

/** Ink bounding box of a PNG, used as the "did anything actually draw" check. */
function inkBox(png) {
  const script = `
import sys, numpy as np
from PIL import Image
im = Image.open(sys.argv[1]).convert("RGB")
a = np.asarray(im).astype(np.int16)
lum = a.max(axis=2)
ys, xs = np.where(lum > 34)
if len(xs) == 0:
    print("0 0 0 0", im.size[0], im.size[1]); raise SystemExit
print(xs.min(), ys.min(), xs.max(), ys.max(), im.size[0], im.size[1])
`;
  const py = process.env.PYTHON || 'C:/Users/Jerry/miniconda3/python.exe';
  const out = execFileSync(py, ['-c', script, png], { encoding: 'utf8' }).trim().split(/\s+/).map(Number);
  return { x0: out[0], y0: out[1], x1: out[2], y1: out[3], w: out[4], h: out[5] };
}

const TARGET_W = 2400;   // wide enough that 26px labels stay crisp on a projector
const chrome = findChrome();
const report = [];

for (const [png, source] of Object.entries(DIAGRAMS)) {
  const { svg } = render(source);
  const [, , vbW, vbH] = svg.match(/viewBox="([\d.\s-]+)"/)[1].split(/\s+/).map(Number);
  const height = Math.round((TARGET_W * vbH) / vbW);
  const stem = png.replace(/\.png$/, '');

  fs.writeFileSync(path.join(OUT, `${stem}.mmd`), source, 'utf8');
  fs.writeFileSync(path.join(OUT, `${stem}.svg`), svg, 'utf8');

  const html = path.join(OUT, `${stem}.html`);
  fs.writeFileSync(html, PAGE(svg, TARGET_W, height), 'utf8');
  const target = path.join(OUT, png);
  execFileSync(chrome, [
    '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-sandbox',
    '--force-device-scale-factor=1.5',
    `--window-size=${TARGET_W},${height}`,
    `--screenshot=${target}`,
    `file:///${html.replace(/\\/g, '/')}`,
  ], { stdio: 'ignore' });
  fs.unlinkSync(html);

  // A blank or clipped export renders "successfully" too, so measure it.
  const box = inkBox(target);
  const fillW = (box.x1 - box.x0) / box.w;
  const fillH = (box.y1 - box.y0) / box.h;
  const ok = fillW > 0.8 && fillH > 0.5;
  report.push({ png, vbW, vbH, px: `${box.w}x${box.h}`,
                ink: `${Math.round(fillW * 100)}% x ${Math.round(fillH * 100)}%`, ok });
}

let bad = 0;
for (const r of report) {
  console.log(`[${r.ok ? 'ok  ' : 'FAIL'}] ${r.png.padEnd(24)} ${r.px.padStart(10)}  ` +
              `内容占幅 ${r.ink}`);
  if (!r.ok) bad += 1;
}
if (bad) {
  console.error(`\n${bad} 张图的绘制区没有铺满画布，先查 Mermaid 源再交付`);
  process.exit(2);
}
