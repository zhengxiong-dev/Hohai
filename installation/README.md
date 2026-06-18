# UAV-YOLO 安装脚本集合

本目录包含UAV-YOLO项目的所有安装脚本和文档，用于在云服务器上一键部署环境。

## 📁 目录结构

```
installation/
├── install.sh              # 完整Bash安装脚本 (推荐)
├── install_env.py          # Python安装脚本 (跨平台)
├── quick_install.sh        # 快速安装脚本 (极简)
├── test_installation.py    # 安装验证测试脚本
├── INSTALL.md              # 详细安装指南
├── DEPLOYMENT.md           # 快速部署参考
├── QUICKSTART.txt          # 快速开始指南
└── README.md               # 本文件
```

## 🚀 快速开始

### 方式1: 完整安装 (推荐)

```bash
cd installation
bash install.sh
```

**特点**:
- ✅ 自动检测GPU和CUDA版本
- ✅ 支持conda或venv环境
- ✅ 完整的错误处理和验证
- ✅ 彩色日志输出
- ⏱️ 时间: 3-5分钟

### 方式2: 快速安装

```bash
cd installation
bash quick_install.sh
```

**特点**:
- ✅ 无交互确认
- ✅ 默认配置
- ⏱️ 时间: 1-2分钟

### 方式3: Python安装

```bash
cd installation
python3 install_env.py
```

**特点**:
- ✅ 纯Python实现
- ✅ 跨平台兼容
- ✅ 更好的错误报告

## 🔧 安装选项

### CPU模式 (无GPU服务器)
```bash
bash install.sh --cpu-only
```

### 指定CUDA版本
```bash
bash install.sh --cuda-version 11.8   # CUDA 11.8
bash install.sh --cuda-version 12.1   # CUDA 12.1
bash install.sh --cuda-version 12.4   # CUDA 12.4
```

### 使用系统Python (不创建conda环境)
```bash
bash install.sh --skip-conda
```

### Python脚本选项
```bash
python3 install_env.py --cpu-only
python3 install_env.py --cuda-version 12.1
```

## ✅ 验证安装

安装完成后，运行验证脚本：

```bash
cd installation
python3 test_installation.py
```

验证内容:
- ✓ 所有依赖包导入
- ✓ CUDA可用性
- ✓ 自定义模块功能
- ✓ 模型加载测试

## 📚 文档说明

### INSTALL.md
完整的安装指南，包含:
- 支持的环境列表
- 详细安装步骤
- 常见问题解决
- 手动安装方案

### DEPLOYMENT.md
快速部署参考卡片，包含:
- 三种安装方式对比
- 云服务器部署流程
- 常用命令速查

### QUICKSTART.txt
纯文本快速指南，适合:
- 命令行查看 (`cat QUICKSTART.txt`)
- 打印参考
- 新手入门

## 🔄 云服务器部署流程

### 第一步: 上传项目
```bash
# 方法1: Git克隆
git clone <your-repo-url>
cd UAV-YOLO/installation

# 方法2: SCP上传
scp -r UAV-YOLO/ user@server:/path/
ssh user@server
cd /path/UAV-YOLO/installation
```

### 第二步: 运行安装
```bash
bash install.sh
```

### 第三步: 激活环境
```bash
# 如果使用conda
conda activate uav-yolo

# 如果使用venv
source ../venv/bin/activate
```

### 第四步: 验证安装
```bash
python3 test_installation.py
```

### 第五步: 开始使用
```bash
cd ..  # 返回项目根目录
python train.py --exp baseline
```

## 🎯 安装脚本对比

| 特性 | install.sh | install_env.py | quick_install.sh |
|------|-----------|----------------|------------------|
| 自动检测GPU/CUDA | ✅ | ✅ | ✅ |
| 自动选择CUDA版本 | ✅ | ✅ | ❌ |
| 支持conda环境 | ✅ | ❌ | ❌ |
| 支持venv环境 | ✅ | ✅ | ❌ |
| 详细日志输出 | ✅ | ✅ | ⚠️ |
| 错误恢复处理 | ✅ | ✅ | ❌ |
| CPU-only选项 | ✅ | ✅ | ✅ |
| 交互式确认 | ✅ | ✅ | ❌ |
| 安装时间 | 3-5分钟 | 3-5分钟 | 1-2分钟 |
| 推荐场景 | 首次部署 | 跨平台 | 快速测试 |

## 🛠️ 自动完成的任务

所有安装脚本都会自动完成:

1. ✓ 检测系统环境 (OS, GPU, CUDA)
2. ✓ 创建Python虚拟环境
3. ✓ 安装PyTorch (匹配CUDA版本)
4. ✓ 安装所有项目依赖
5. ✓ 配置自定义模块到ultralytics
6. ✓ 运行完整性验证测试
7. ✓ 生成安装报告

## 📋 支持的环境

### 操作系统
- ✅ Ubuntu 18.04+
- ✅ CentOS 7+
- ✅ Debian 10+
- ✅ 其他主流Linux发行版

### Python版本
- ✅ Python 3.8 (最低)
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11 (推荐)

### CUDA版本
- ✅ CUDA 11.8
- ✅ CUDA 12.1
- ✅ CUDA 12.4
- ✅ CPU-only (无CUDA)

### GPU支持
- ✅ NVIDIA RTX系列 (3090, 4090, etc.)
- ✅ NVIDIA Tesla系列 (V100, A100, etc.)
- ✅ NVIDIA GTX系列 (1080Ti, etc.)

## 🆘 故障排除

### 问题1: CUDA不可用
```bash
# 检查GPU
nvidia-smi

# 检查CUDA
nvcc --version

# 重新安装指定版本
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
chmod +x install.sh install_env.py quick_install.sh test_installation.py
```

### 问题4: pip下载慢
```bash
# 使用国内镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
bash install.sh
```

## 💡 使用建议

### 推荐配置
- **训练**: GPU (RTX 3090+), 24GB+ VRAM
- **评估**: GPU (GTX 1080Ti+), 11GB+ VRAM  
- **推理**: CPU可用, GPU更快

### 性能优化
- CPU服务器: 不推荐用于训练 (极慢)
- GPU服务器: 确保CUDA正确安装
- 网络优化: 配置pip镜像加速下载

## 📖 下一步

安装完成后:

1. **返回项目根目录**
   ```bash
   cd ..
   ```

2. **查看训练选项**
   ```bash
   python train.py --help
   ```

3. **准备数据集** (见项目README.md)
   ```bash
   python tools/convert_visdrone_to_yolo.py
   ```

4. **开始训练**
   ```bash
   python train.py --exp baseline
   python train.py --exp uav_yolo_pt
   ```

5. **评估模型**
   ```bash
   python tools/eval_scale_ap.py --weights runs/detect/*/weights/best.pt
   ```

## 📞 技术支持

如遇问题:
1. 查看详细文档: `cat INSTALL.md`
2. 运行验证脚本: `python3 test_installation.py`
3. 检查安装日志中的错误信息
4. 尝试使用 `--cpu-only` 模式测试

---

**提示**: 首次运行会下载预训练权重 (约20MB)，请确保网络连接正常。

**版本**: v1.0  
**更新日期**: 2026-06-18
