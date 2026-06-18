# 快速安装 UAV-YOLO 环境

## 📦 一键安装

所有安装脚本和文档都在 `installation/` 目录中。

```bash
cd installation
bash install.sh
```

## 📁 安装文件位置

```
installation/
├── install.sh           # 完整安装脚本 (推荐)
├── quick_install.sh     # 快速安装
├── install_env.py       # Python安装脚本
├── test_installation.py # 验证脚本
└── README.md            # 详细说明
```

## 🚀 三种安装方式

### 1. 完整安装 (推荐)
```bash
cd installation && bash install.sh
```

### 2. 快速安装
```bash
cd installation && bash quick_install.sh
```

### 3. Python安装
```bash
cd installation && python3 install_env.py
```

## ✅ 验证安装

```bash
cd installation
python3 test_installation.py
```

## 📚 查看完整文档

```bash
cd installation
cat README.md          # 完整安装指南
cat INSTALL.md         # 详细安装文档
cat QUICKSTART.txt     # 快速开始
```

## 💡 常用选项

```bash
# CPU模式
bash install.sh --cpu-only

# 指定CUDA版本
bash install.sh --cuda-version 12.1

# 使用系统Python
bash install.sh --skip-conda
```

---

详细信息请查看 `installation/README.md`
