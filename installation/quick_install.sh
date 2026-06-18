#!/bin/bash

###############################################################################
# Quick Start Script - 最简安装
# 适合熟悉环境配置的用户，跳过所有交互式确认
###############################################################################

set -e

echo "🚀 UAV-YOLO Quick Install"
echo "=========================="

# 检测CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected"
    HAS_GPU=true
else
    echo "✓ CPU mode"
    HAS_GPU=false
fi

# 使用Python3
PYTHON=python3

# 升级pip
echo "📦 Installing dependencies..."
$PYTHON -m pip install --upgrade pip -q

# 安装PyTorch
if [ "$HAS_GPU" = true ]; then
    # 自动选择CUDA 12.1 (兼容性最好)
    $PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
else
    $PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
fi

# 安装依赖
$PYTHON -m pip install -r requirements.txt -q

# 配置自定义模块
echo "⚙️  Setting up custom modules..."
$PYTHON setup_env.py

# 验证
echo "✅ Verifying..."
$PYTHON -c "import torch; from ultralytics.nn.modules.custom import EMA; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Usage:"
echo "  python train.py --exp baseline"
echo "  python train.py --exp uav_yolo_pt"
