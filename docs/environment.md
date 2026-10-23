# 环境搭建（uv）

`python` 打到的是系统 Python 3.11，它装了 `torch` 但没装 `gradio`，所以
`python -m app.main` 一进来就 `ModuleNotFoundError: No module named 'gradio'`。
本仓库的东西全部跑在 3.12 + CUDA 版 torch 上，用 uv 建一个仓库内的虚拟环境最省事。

## 一条命令

```bash
cd /d/projects/vpteam/GLIDE-Net

uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121
export UV_LINK_MODE=copy          # 跨盘符，缓存在 C，venv 在 D
uv pip install --python .venv/Scripts/python.exe -r requirements.txt -r requirements-app.txt
```

`requirements-app.txt` 里已经有 `python-pptx`？没有，幻灯片自检要它，单独补：

```bash
uv pip install --python .venv/Scripts/python.exe python-pptx defusedxml lxml matplotlib
```

## 为什么 torch 要单独装

`requirements.txt` 里写的是 `torch>=2.4`，从 PyPI 装会拿到 CPU 轮子。CUDA 版只在
`download.pytorch.org` 上，所以要先用 `--index-url` 装 torch 和 torchvision，再装
其余依赖。顺序反过来的话，后面装 `ultralytics` 时 pip 会觉得 torch 已满足，不会
把它换成 CUDA 版。

## 验证

```bash
.venv/Scripts/python.exe -c "import torch;print(torch.__version__, torch.cuda.is_available())"
# 2.5.1+cu121 True

.venv/Scripts/python.exe -m pytest tests -q          # 27 passed
.venv/Scripts/python.exe tools/verify.py             # 40 项全过
```

## 启动

```bash
.venv/Scripts/python.exe -m app.main --port 7860 \
    --checkpoint ckpt/glide_net_6440647.pth \
    --detector runs/detect/detection/runs/neudet_yolov8m/weights/best.pt
```

四点五秒起服务，浏览器开 `http://127.0.0.1:7860`。

## 本机实测的版本

| 包 | 版本 |
|----|------|
| Python | 3.12.9（uv 下载的独立解释器） |
| torch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 |
| gradio | 6.30.0 |
| ultralytics | 8.4.184 |
| onnx / onnxruntime-gpu | 1.23.2 / 1.22.0 |
| opencv-python | 5.0.0 |
| numpy / scipy | 2.5.2 / 1.18.1 |

`.venv` 占了 5.5 GB，已被 `.gitignore` 挡住，不会进仓库。

## 两个坑

**`--device` 默认是 `cuda`。** 没有 GPU 的机器要在命令后面加 `--device cpu`，否则
`build_pipeline` 会在 torch 那步报错。

**空目录 `dist/` 和 `app/static/_probe/` 不要留。** 调试上传时用过的临时目录记得删，
`git status` 里出现它们说明还没清干净。
