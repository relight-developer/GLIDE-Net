"""Visual system for the inspection workstation (Gradio CSS).

The world is the steel mill inspection gantry and the engineering drawing
sheet: near-black steel, hairline rules, safety yellow in the title block,
condensed engineering lettering for headings and measurements set in mono.
Nothing decorative is added that an inspector would not read.
"""
from __future__ import annotations

import os

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "static", "fonts")

CSS = """
@font-face{font-family:'Barlow Condensed';src:url('/gradio_api/file=__FONTS__/BarlowCondensed-SemiBold.ttf') format('truetype');font-weight:600;font-display:swap;}
@font-face{font-family:'Inter';src:url('/gradio_api/file=__FONTS__/Inter-Regular.ttf') format('truetype');font-weight:400;font-display:swap;}
@font-face{font-family:'Inter';src:url('/gradio_api/file=__FONTS__/Inter-Medium.ttf') format('truetype');font-weight:500;font-display:swap;}
@font-face{font-family:'JetBrains Mono';src:url('/gradio_api/file=__FONTS__/JetBrainsMono-Regular.ttf') format('truetype');font-weight:400;font-display:swap;}

:root{
  --ink:#0B0F14; --steel-900:#0E141B; --steel-800:#121A23; --steel-700:#18222D;
  --rule:#22303D; --rule-soft:#1A242F;
  --paper:#E8EEF4; --muted:#93A2B1; --muted-dim:#8695A4;
  --safety:#F5C542; --safety-dim:#8A6F1E;
  --sev:#E5484D; --med:#F5A524; --light:#4C9AFF; --ok:#30A46C;
}

body,.gradio-container{background:var(--ink)!important;color:var(--paper)!important;
  font-family:'Inter','Microsoft YaHei',system-ui,sans-serif!important;}
.gradio-container{max-width:1560px!important;padding:0 26px 64px!important;}
footer{display:none!important;}
*::selection{background:var(--safety);color:#0B0F14;}
::-webkit-scrollbar{width:10px;height:10px;}
::-webkit-scrollbar-track{background:var(--steel-900);}
::-webkit-scrollbar-thumb{background:#2A3947;border-radius:0;}
::-webkit-scrollbar-thumb:hover{background:#3A4C5E;}
:focus-visible{outline:2px solid var(--safety)!important;outline-offset:2px!important;}

/* ---- title block: the drawing sheet header ---- */
#titleblock{display:grid;grid-template-columns:minmax(0,1.65fr) repeat(3,minmax(120px,.5fr));
  border-bottom:2px solid var(--safety);margin:0 0 22px;background:
  repeating-linear-gradient(90deg,var(--steel-800) 0 1px,transparent 1px 26px);}
#titleblock .cell{padding:18px 20px;border-left:1px solid var(--rule);min-width:0;}
#titleblock .cell:first-child{border-left:none;padding-left:0;}
#titleblock .k{font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--muted-dim);margin-bottom:6px;}
#titleblock .v{font-family:'Barlow Condensed',sans-serif;font-weight:600;font-size:27px;
  line-height:1.05;letter-spacing:.006em;color:var(--paper);}
#titleblock .v.small{font-size:19px;color:#C7D3DE;}
#titleblock .spec{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--muted);
  margin-top:7px;letter-spacing:.02em;}
#titleblock h1{font-family:'Barlow Condensed',sans-serif;font-weight:600;font-size:44px;
  line-height:1;margin:0;letter-spacing:-.01em;color:#fff;}
#titleblock h1 em{font-style:normal;color:var(--safety);}

/* ---- generic surface ---- */
.pt-panel{background:var(--steel-800);border:1px solid var(--rule);border-radius:0;padding:16px;}
.pt-label{font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.17em;
  text-transform:uppercase;color:var(--muted-dim);}
.rule{height:1px;background:var(--rule);margin:18px 0;}

/* tabs read as an instrument selector strip */
.tab-nav,div[role="tablist"]{border-bottom:1px solid var(--rule)!important;gap:2px!important;}
button[role="tab"]{font-family:'Barlow Condensed',sans-serif!important;font-size:16px!important;
  letter-spacing:.05em!important;text-transform:uppercase;color:var(--muted)!important;
  border:none!important;border-bottom:2px solid transparent!important;border-radius:0!important;
  padding:9px 16px!important;background:transparent!important;}
button[role="tab"]:hover{color:var(--paper)!important;background:var(--steel-800)!important;}
button[role="tab"][aria-selected="true"]{color:var(--safety)!important;
  border-bottom-color:var(--safety)!important;background:transparent!important;}

/* buttons */
button.primary{background:var(--safety)!important;color:#0B0F14!important;border:none!important;
  border-radius:0!important;font-family:'Barlow Condensed',sans-serif!important;
  font-size:16px!important;letter-spacing:.07em;text-transform:uppercase;font-weight:600!important;
  box-shadow:none!important;transition:background .16s cubic-bezier(.2,.8,.3,1);}
button.primary:hover{background:#FFD65C!important;}
button.secondary{background:transparent!important;color:var(--paper)!important;
  border:1px solid var(--rule)!important;border-radius:0!important;
  font-family:'Barlow Condensed',sans-serif!important;font-size:15px!important;
  letter-spacing:.06em;text-transform:uppercase;}
button.secondary:hover{border-color:var(--safety)!important;color:var(--safety)!important;}

/* inputs */
input[type=text],input[type=number],textarea,select{background:var(--steel-900)!important;
  border:1px solid var(--rule)!important;border-radius:0!important;color:var(--paper)!important;
  font-family:'JetBrains Mono',monospace!important;font-size:12.5px!important;}
input[type=range]{accent-color:var(--safety)!important;}
label span,.block-info{color:var(--muted)!important;font-family:'JetBrains Mono',monospace!important;
  font-size:11px!important;letter-spacing:.06em;}

/* image drop zones */
.pt-frame{border:1px solid var(--rule)!important;border-radius:0!important;
  background:var(--steel-900)!important;}
.image-container,.image-frame{background:var(--steel-900)!important;border-radius:0!important;}
.upload-container{border:1px dashed #2E3D4C!important;border-radius:0!important;
  background:var(--steel-900)!important;}

/* tables */
table{border-collapse:collapse!important;font-family:'JetBrains Mono',monospace!important;
  font-size:12px!important;}
thead th{background:var(--steel-700)!important;color:var(--muted)!important;
  font-family:'Barlow Condensed',sans-serif!important;font-size:13px!important;
  letter-spacing:.09em;text-transform:uppercase;font-weight:600!important;
  border-bottom:1px solid var(--rule)!important;}
tbody td{border-bottom:1px solid var(--rule-soft)!important;color:var(--paper)!important;}
tbody tr:hover td{background:var(--steel-700)!important;}
.dataframe{background:transparent!important;}

/* report block rendered as a drawing's inspection sheet */
#report{font-family:'Inter',sans-serif;}
#report .head{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;
  padding-bottom:10px;border-bottom:1px solid var(--rule);margin-bottom:14px;}
#report .verdict{font-family:'Barlow Condensed',sans-serif;font-size:38px;font-weight:600;
  line-height:1;letter-spacing:-.01em;}
#report .verdict.severe{color:var(--sev);} #report .verdict.medium{color:var(--med);}
#report .verdict.light{color:var(--light);} #report .verdict.none{color:var(--ok);}
#report .sub{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--muted);
  letter-spacing:.04em;}
#report .gauges{display:grid;grid-template-columns:repeat(auto-fit,minmax(122px,1fr));
  gap:1px;background:var(--rule);border:1px solid var(--rule);margin:0 0 16px;}
#report .gauge{background:var(--steel-800);padding:11px 13px;}
#report .gauge .n{font-family:'Barlow Condensed',sans-serif;font-size:26px;font-weight:600;
  line-height:1.1;color:#fff;font-variant-numeric:tabular-nums;}
#report .gauge .n small{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);
  margin-left:3px;letter-spacing:.02em;}
#report .gauge .k{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.15em;
  text-transform:uppercase;color:var(--muted-dim);margin-top:3px;}
#report .row{display:grid;grid-template-columns:20px 1fr 62px 74px;gap:10px;align-items:center;
  padding:7px 0;border-bottom:1px solid var(--rule-soft);font-size:12.5px;}
#report .row .tag{font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--muted-dim);}
#report .row .score{font-family:'JetBrains Mono',monospace;font-size:12px;text-align:right;
  color:#fff;font-variant-numeric:tabular-nums;}
#report .sev{font-family:'Barlow Condensed',sans-serif;font-size:12.5px;letter-spacing:.07em;
  text-transform:uppercase;text-align:right;}
#report .sev.severe{color:var(--sev);} #report .sev.medium{color:var(--med);}
#report .sev.light{color:var(--light);}
#report .empty{color:var(--muted-dim);font-size:12.5px;padding:8px 0;}
#report .note{font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--muted-dim);
  letter-spacing:.02em;margin-top:12px;line-height:1.6;}

/* metric tiles used by the model tab */
#metrics .tile{margin-bottom:18px;}
#metrics .tile h3{font-family:'Barlow Condensed',sans-serif;font-size:20px;font-weight:600;
  letter-spacing:.02em;margin:0 0 10px;color:var(--paper);}
#metrics .tile h3 span{font-family:'JetBrains Mono',monospace;font-size:10.5px;
  color:var(--muted-dim);letter-spacing:.14em;text-transform:uppercase;margin-left:9px;}

/* prose in the docs tab */
#prose{color:#C7D3DE;font-size:13.5px;line-height:1.75;max-width:78ch;}
#prose h2{font-family:'Barlow Condensed',sans-serif;font-size:24px;font-weight:600;
  color:#fff;margin:22px 0 8px;letter-spacing:.01em;}
#prose h2:first-child{margin-top:0;}
#prose code{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--safety);
  background:var(--steel-700);padding:1px 5px;}
#prose ul{padding-left:18px;} #prose li{margin:4px 0;}
#prose .kbd{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);}

/* status line */
#status{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--muted);
  letter-spacing:.03em;border-left:2px solid var(--safety);padding:7px 0 7px 11px;
  background:var(--steel-900);}
#status b{color:var(--safety);font-weight:400;}
#status.err{border-left-color:var(--sev);} #status.err b{color:var(--sev);}
"""


def css() -> str:
    """CSS with the local font folder substituted to an absolute path."""
    fonts = FONT_DIR.replace("\\", "/")
    return CSS.replace("__FONTS__", fonts)
