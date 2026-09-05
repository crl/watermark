# WaterMark

Electron 桌面应用：在视频上框选固定区域，用 **ProPainter** 做 AI 修复，或回退到 FFmpeg `delogo`。

请只处理你拥有或已获授权的素材。

## 环境

- Windows + NVIDIA 显卡（AI 模式）
- Node.js 20+
- Python 3.11
- CUDA 版 PyTorch（仅 AI 模式需要）

## 安装

```bat
cd electron
npm install

cd ..\backend
python -m pip install -r requirements.txt
```

FFmpeg 会由 `imageio-ffmpeg` 自动提供，不必单独安装。

### AI 模式（ProPainter）

1. 安装 CUDA 版 PyTorch，例如：

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-ai.txt
```

2. 下载源码（也可在应用里点「下载 ProPainter 源码」）：

```bat
python setup_propainter.py
```

权重会在第一次推理时自动下载到 `backend/vendor/ProPainter/weights/`。

## 开发

在仓库根目录：

```bat
npm run dev
```

窗口打开后：选择视频 → 拖出矩形 → 开始修复。没有 ProPainter 时会自动用 FFmpeg 邻域修补，方便先打通流程。

## 打包

```bat
cd electron
$env:ELECTRON_MIRROR="https://npmmirror.com/mirrors/electron/"
$env:ELECTRON_BUILDER_BINARIES_MIRROR="https://npmmirror.com/mirrors/electron-builder-binaries/"
npm run build:win
```

安装包在 `electron/release/`。包内只含界面和 Python 脚本，**不含** PyTorch / CUDA / 模型。目标机器需要本机 Python 环境，或设置 `WATERMARK_PYTHON` 指向解释器。

## 目录

- `electron/` 主进程、预加载、React 界面
- `backend/` FastAPI sidecar、FFmpeg 管线、ProPainter 调用
