# UAV-YOLO 一键安装指南

本项目提供两种一键安装方式，适用于不同的云服务器环境。

## 安装方式选择

### 方式1: Bash脚本安装 (推荐)

适用于 Linux 服务器，功能最完整。

```bash
# 基本安装 (自动检测GPU和CUDA)
bash install.sh

# CPU版本 (无GPU服务器)
bash install.sh --cpu-only

# 指定CUDA版本
bash install.sh --cuda-version 12.1

# 使用系统Python而非conda
bash install.sh --skip-conda
```

### 方式2: Python脚本安装

纯Python实现，跨平台兼容性更好。

```bash
# 基本安装
python3 install_env.py

# CPU版本
python3 install_env.py --cpu-only

# 指定CUDA版本
python3 install_env.py --cuda-version 12.1
```

## 云服务器快速部署步骤

### 1. 上传项目到服务器

```bash
# 方法1: 使用git clone
git clone <your-repo-url>
cd UAV-YOLO

# 方法2: 使用scp上传
scp -r UAV-YOLO/ user@server:/path/to/destination/
```

### 2. 运行安装脚本

```bash
cd UAV-YOLO

# GPU服务器
bash install.sh

# 或使用Python版本
python3 install_env.py
```

### 3. 激活环境

```bash
# 如果使用conda (默认)
conda activate uav-yolo

# 如果使用venv (--skip-conda模式)
source venv/bin/activate
```

### 4. 验证安装

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 常见云平台配置

### 阿里云/腾讯云 GPU实例

```bash
# 通常已有CUDA驱动,直接运行
bash install.sh
```

### AWS/GCP GPU实例

```bash
# Deep Learning AMI 通常预装CUDA
bash install.sh
```

### AutoDL/矩池云

```bash
# 这些平台通常预装PyTorch环境
# 可以只安装项目依赖
pip install -r requirements.txt
python setup_env.py
```

### CPU-only 服务器

```bash
bash install.sh --cpu-only
```

## 支持的环境

### 操作系统
- ✅ Ubuntu 18.04+
- ✅ CentOS 7+
- ✅ Debian 10+
- ✅ 其他主流Linux发行版

### Python版本
- ✅ Python 3.8 (最低要求)
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11 (推荐)

### CUDA版本
- ✅ CUDA 11.8
- ✅ CUDA 12.1
- ✅ CUDA 12.4
- ✅ CPU-only (无CUDA)

### GPU支持
- ✅ NVIDIA RTX 系列 (3090, 4090, etc.)
- ✅ NVIDIA Tesla 系列 (V100, A100, etc.)
- ✅ NVIDIA GTX 系列 (1080Ti, etc.)

## 安装脚本功能

自动完成以下步骤:

1. ✅ 检测系统环境 (OS, GPU, CUDA)
2. ✅ 创建Python虚拟环境 (conda或venv)
3. ✅ 安装匹配的PyTorch版本
4. ✅ 安装所有项目依赖
5. ✅ 配置自定义模块到ultralytics
6. ✅ 运行验证测试
7. ✅ 提供详细的安装报告

## 故障排除

### 问题1: CUDA不可用

```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查CUDA版本
nvcc --version

# 重装PyTorch
pip uninstall torch torchvision
bash install.sh --cuda-version 12.1
```

### 问题2: conda命令不存在

```bash
# 使用系统Python模式
bash install.sh --skip-conda

# 或安装miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 问题3: 权限不足

```bash
# 添加执行权限
chmod +x install.sh
chmod +x install_env.py

# 如果需要sudo
sudo bash install.sh
```

### 问题4: 依赖冲突

```bash
# 删除旧环境重新安装
conda env remove -n uav-yolo
bash install.sh
```

## 手动安装 (备选方案)

如果自动安装脚本失败，可以手动执行:

```bash
# 1. 创建环境
conda create -n uav-yolo python=3.11 -y
conda activate uav-yolo

# 2. 安装PyTorch (根据CUDA版本选择)
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置自定义模块
python setup_env.py

# 5. 验证
python -c "from ultralytics.nn.modules.custom import EMA; print('Success!')"
```

## 性能优化建议

### GPU服务器
- 推荐: RTX 4090 / A100 (24GB+ VRAM)
- 最低: GTX 1080Ti (11GB VRAM)

### CPU服务器
- 不推荐用于训练 (速度极慢)
- 可用于推理和评估
- 建议: 16+ CPU核心, 32GB+ RAM

### 数据集准备
```bash
# 下载数据集后放在这个目录
mkdir -p datasets/

# 转换格式
python tools/convert_visdrone_to_yolo.py
python tools/convert_nwpu_to_yolo.py
python prepare_aitod.py
```

## 下一步

安装完成后:

```bash
# 1. 检查环境
python train.py --help

# 2. 准备数据集 (见上方)

# 3. 开始训练
python train.py --exp baseline          # 基线模型
python train.py --exp uav_yolo_pt       # UAV-YOLO完整模型

# 4. 评估模型
python tools/eval_scale_ap.py --weights runs/detect/*/weights/best.pt

# 5. 测试推理
python visualize.py --weights runs/detect/*/weights/best.pt --source test_image.jpg
```

## 技术支持

如遇问题:
1. 查看安装日志中的错误信息
2. 确认GPU驱动和CUDA版本匹配
3. 尝试使用 `--cpu-only` 模式测试
4. 检查防火墙是否阻止pip下载

---

**提示**: 首次运行会下载预训练权重 (约20MB)，请确保网络连接正常。
