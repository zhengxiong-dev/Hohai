# 云服务器部署快速参考

## 📦 三种安装方式

### 1️⃣ 完整安装 (推荐)
```bash
bash install.sh
```
- ✅ 自动检测GPU/CUDA
- ✅ 支持conda和venv
- ✅ 详细日志和验证
- ✅ 支持多种配置选项

### 2️⃣ Python安装 (跨平台)
```bash
python3 install_env.py
```
- ✅ 纯Python实现
- ✅ 更好的错误处理
- ✅ 跨平台兼容

### 3️⃣ 快速安装 (极简)
```bash
bash quick_install.sh
```
- ✅ 无交互确认
- ✅ 一分钟完成
- ✅ 适合快速测试

## 🚀 云服务器一键部署

```bash
# 1. 上传代码到服务器
scp -r UAV-YOLO/ user@server:/path/

# 2. SSH登录
ssh user@server

# 3. 进入目录并安装
cd /path/UAV-YOLO
bash install.sh

# 4. 激活环境 (如果使用conda)
conda activate uav-yolo

# 5. 开始训练
python train.py --exp baseline
```

## 📋 安装选项

| 选项 | install.sh | install_env.py | quick_install.sh |
|------|-----------|----------------|------------------|
| 完整功能 | ✅ | ✅ | ⚠️ 基础 |
| GPU检测 | ✅ | ✅ | ✅ |
| CUDA版本选择 | ✅ | ✅ | ❌ (默认12.1) |
| conda支持 | ✅ | ❌ | ❌ |
| 详细日志 | ✅ | ✅ | ⚠️ 简化 |
| 交互式确认 | ✅ | ✅ | ❌ |

## 🔧 常用命令

```bash
# CPU模式安装
bash install.sh --cpu-only

# 指定CUDA版本
bash install.sh --cuda-version 12.1

# 使用系统Python
bash install.sh --skip-conda

# Python版本
python3 install_env.py --cuda-version 12.4
```

## ✅ 验证安装

```bash
python -c "
import torch
from ultralytics.nn.modules.custom import EMA
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('Custom modules: OK')
"
```

## 📖 详细文档

- 完整安装指南: `INSTALL.md`
- 项目说明: `README.md`
- 训练脚本: `train.py --help`

## 🐛 常见问题

**Q: CUDA不可用?**
```bash
nvidia-smi  # 检查GPU
nvcc --version  # 检查CUDA
bash install.sh --cuda-version 12.1  # 重装
```

**Q: pip下载慢?**
```bash
# 使用国内镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
bash install.sh
```

**Q: 没有conda?**
```bash
bash install.sh --skip-conda
# 或
python3 install_env.py
```

## 💡 性能建议

| 用途 | 推荐配置 |
|------|---------|
| 训练 | GPU (RTX 3090+), 24GB+ VRAM |
| 评估 | GPU (GTX 1080Ti+), 11GB+ VRAM |
| 推理 | CPU可用, GPU更快 |

## 🎯 下一步

```bash
# 1. 准备数据集
python tools/convert_visdrone_to_yolo.py

# 2. 训练基线
python train.py --exp baseline

# 3. 训练UAV-YOLO
python train.py --exp uav_yolo_pt

# 4. 评估
python tools/eval_scale_ap.py --weights runs/detect/*/weights/best.pt
```
