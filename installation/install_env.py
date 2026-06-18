#!/usr/bin/env python3
"""
UAV-YOLO Python-based Installation Script

A pure Python alternative to install.sh for better cross-platform compatibility.

Usage:
    python install_env.py [--cpu-only] [--cuda-version 11.8|12.1|12.4]
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


class Colors:
    """ANSI color codes"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def log_info(msg):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")


def log_success(msg):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")


def log_warn(msg):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")


def log_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


def run_command(cmd, check=True, capture=False):
    """Run shell command and return result"""
    try:
        if capture:
            result = subprocess.run(
                cmd, shell=True, check=check,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True
            )
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check)
            return None
    except subprocess.CalledProcessError as e:
        if check:
            log_error(f"Command failed: {cmd}")
            log_error(f"Error: {e}")
            sys.exit(1)
        return None


def check_gpu():
    """Check if NVIDIA GPU is available"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')[0]
            log_success(f"NVIDIA GPU detected: {gpu_info}")
            return True
    except FileNotFoundError:
        pass
    log_warn("No NVIDIA GPU detected")
    return False


def detect_cuda_version():
    """Auto-detect CUDA version"""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False
        )
        if result.returncode == 0:
            # Parse version from output
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    import re
                    match = re.search(r'release (\d+\.\d+)', line)
                    if match:
                        version = match.group(1)
                        log_info(f"Detected CUDA version: {version}")

                        # Map to PyTorch-supported versions
                        major = int(version.split('.')[0])
                        if major == 11:
                            return "11.8"
                        elif major == 12:
                            minor = int(version.split('.')[1])
                            if minor <= 1:
                                return "12.1"
                            else:
                                return "12.4"
    except FileNotFoundError:
        log_warn("nvcc not found, cannot detect CUDA version")

    return "12.1"  # Default


def get_pytorch_install_command(cuda_version=None, cpu_only=False):
    """Generate PyTorch installation command"""
    if cpu_only:
        return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"

    cuda_urls = {
        "11.8": "https://download.pytorch.org/whl/cu118",
        "12.1": "https://download.pytorch.org/whl/cu121",
        "12.4": "https://download.pytorch.org/whl/cu124",
    }

    if cuda_version not in cuda_urls:
        log_error(f"Unsupported CUDA version: {cuda_version}")
        sys.exit(1)

    return f"pip install torch torchvision --index-url {cuda_urls[cuda_version]}"


def verify_installation():
    """Verify that everything is installed correctly"""
    log_info("Verifying installation...")

    verify_code = """
import sys
import torch
import torchvision
import ultralytics

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"Ultralytics: {ultralytics.__version__}")

if torch.cuda.is_available():
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA: Not available (CPU mode)")

# Test custom modules
try:
    from ultralytics.nn.modules.custom import EMA, BiFPN_Concat, SGFA, SOA_SAFM
    print("Custom modules: Imported successfully")

    # Quick functional test
    x = torch.randn(1, 256, 32, 32)
    ema = EMA(256)
    y = ema(x)
    assert y.shape == x.shape
    print("Functional test: PASSED")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

print("\\n✅ All checks passed!")
"""

    try:
        subprocess.run(
            [sys.executable, "-c", verify_code],
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        log_error("Verification failed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="UAV-YOLO Environment Installation"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Install CPU-only version (no GPU support)"
    )
    parser.add_argument(
        "--cuda-version",
        choices=["11.8", "12.1", "12.4"],
        help="CUDA version to use (default: auto-detect)"
    )
    args = parser.parse_args()

    # Banner
    print("=" * 60)
    print("  UAV-YOLO Environment Installation")
    print("  Python-based installer")
    print("=" * 60)
    print()

    # Check system
    log_info(f"OS: {platform.system()} {platform.release()}")
    log_info(f"Python: {sys.version.split()[0]}")

    # Check Python version
    if sys.version_info < (3, 8):
        log_error(f"Python {sys.version_info.major}.{sys.version_info.minor} is too old")
        log_error("Please use Python >= 3.8 (recommended: 3.11)")
        sys.exit(1)

    # Determine CPU/GPU mode
    cpu_only = args.cpu_only
    if not cpu_only:
        has_gpu = check_gpu()
        if not has_gpu:
            log_warn("No GPU detected, switching to CPU-only mode")
            cpu_only = True

    # Determine CUDA version
    cuda_version = args.cuda_version
    if not cpu_only and not cuda_version:
        cuda_version = detect_cuda_version()
        log_info(f"Using CUDA version: {cuda_version}")

    # Upgrade pip
    log_info("Upgrading pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip setuptools wheel")

    # Install PyTorch
    log_info("Installing PyTorch...")
    pytorch_cmd = get_pytorch_install_command(cuda_version, cpu_only)
    run_command(pytorch_cmd)

    # Verify PyTorch
    log_info("Verifying PyTorch installation...")
    torch_version = run_command(
        f"{sys.executable} -c 'import torch; print(torch.__version__)'",
        capture=True
    )
    log_success(f"PyTorch {torch_version} installed")

    if not cpu_only:
        cuda_available = run_command(
            f"{sys.executable} -c 'import torch; print(torch.cuda.is_available())'",
            capture=True
        )
        if cuda_available == "True":
            log_success("CUDA is available in PyTorch")
        else:
            log_warn("CUDA is not available (but was expected)")

    # Install project dependencies
    log_info("Installing project dependencies...")
    project_dir = Path(__file__).parent
    requirements = project_dir / "requirements.txt"

    if requirements.exists():
        run_command(f"{sys.executable} -m pip install -r {requirements}")
    else:
        log_error(f"requirements.txt not found at {requirements}")
        sys.exit(1)

    # Setup custom modules
    log_info("Setting up custom modules...")
    setup_script = project_dir / "setup_env.py"

    if setup_script.exists():
        run_command(f"{sys.executable} {setup_script}")
        log_success("Custom modules installed")
    else:
        log_error(f"setup_env.py not found at {setup_script}")
        sys.exit(1)

    # Final verification
    if verify_installation():
        log_success("Installation completed successfully!")

        # Print next steps
        print()
        print("=" * 60)
        print("  Installation Complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print()
        print("1. Prepare datasets (see README.md):")
        print("   - Download VisDrone2019, AI-TOD, or NWPU")
        print("   - Run conversion scripts in tools/")
        print()
        print("2. Start training:")
        print("   python train.py --exp baseline")
        print("   python train.py --exp uav_yolo_pt")
        print()
        if cpu_only:
            log_warn("Note: CPU-only mode will be VERY slow for training")
            log_warn("Consider using a GPU instance for production training")
        print()
    else:
        log_error("Installation verification failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
