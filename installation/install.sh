#!/bin/bash

###############################################################################
# UAV-YOLO One-Click Installation Script
#
# This script will:
# 1. Check system requirements (OS, GPU, CUDA)
# 2. Install Python 3.11 (via conda/mamba if available, or system package)
# 3. Create virtual environment
# 4. Install PyTorch with correct CUDA version
# 5. Install all project dependencies
# 6. Setup custom modules in ultralytics
# 7. Verify installation
#
# Usage:
#   bash install.sh [--cpu-only] [--cuda-version 11.8|12.1|12.4]
#
# Options:
#   --cpu-only         Install CPU-only version of PyTorch (no GPU support)
#   --cuda-version     Specify CUDA version (default: auto-detect)
#   --skip-conda       Skip conda/mamba and use system Python + venv
###############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
CPU_ONLY=false
CUDA_VERSION=""
SKIP_CONDA=false
PYTHON_VERSION="3.11"
ENV_NAME="uav-yolo"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --skip-conda)
            SKIP_CONDA=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo "============================================================"
echo "  UAV-YOLO Environment Installation"
echo "  Automated setup for training and inference"
echo "============================================================"
echo ""

# Step 1: Check system and CUDA
log_info "Checking system requirements..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    log_success "OS: Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    log_warn "OS: macOS (GPU training not available)"
    CPU_ONLY=true
else
    log_error "Unsupported OS: $OSTYPE"
    exit 1
fi

# Check GPU and CUDA
if [ "$CPU_ONLY" = false ]; then
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1

        # Auto-detect CUDA version if not specified
        if [ -z "$CUDA_VERSION" ]; then
            if command -v nvcc &> /dev/null; then
                DETECTED_CUDA=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
                log_info "Detected CUDA version: $DETECTED_CUDA"

                # Map to PyTorch-supported CUDA versions
                if [[ "$DETECTED_CUDA" == 11.* ]]; then
                    CUDA_VERSION="11.8"
                elif [[ "$DETECTED_CUDA" == 12.1* ]] || [[ "$DETECTED_CUDA" == 12.0* ]]; then
                    CUDA_VERSION="12.1"
                elif [[ "$DETECTED_CUDA" == 12.4* ]] || [[ "$DETECTED_CUDA" == 12.[2-9]* ]]; then
                    CUDA_VERSION="12.4"
                else
                    log_warn "CUDA $DETECTED_CUDA detected, defaulting to CUDA 12.1"
                    CUDA_VERSION="12.1"
                fi
            else
                log_warn "nvcc not found, defaulting to CUDA 12.1"
                CUDA_VERSION="12.1"
            fi
        fi
        log_info "Using CUDA version: $CUDA_VERSION for PyTorch installation"
    else
        log_warn "No NVIDIA GPU detected, installing CPU-only version"
        CPU_ONLY=true
    fi
else
    log_info "CPU-only installation requested"
fi

# Step 2: Setup Python environment
log_info "Setting up Python environment..."

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Check for conda/mamba
if [ "$SKIP_CONDA" = false ]; then
    if command -v mamba &> /dev/null; then
        CONDA_CMD="mamba"
        log_success "Found mamba (faster than conda)"
    elif command -v conda &> /dev/null; then
        CONDA_CMD="conda"
        log_success "Found conda"
    else
        log_warn "Conda/mamba not found, will use system Python + venv"
        SKIP_CONDA=true
    fi
fi

# Create environment
if [ "$SKIP_CONDA" = false ]; then
    log_info "Creating conda environment: $ENV_NAME"

    # Check if environment already exists
    if $CONDA_CMD env list | grep -q "^$ENV_NAME "; then
        log_warn "Environment '$ENV_NAME' already exists"
        read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $CONDA_CMD env remove -n "$ENV_NAME" -y
        else
            log_info "Using existing environment"
        fi
    fi

    # Create new environment if it doesn't exist
    if ! $CONDA_CMD env list | grep -q "^$ENV_NAME "; then
        $CONDA_CMD create -n "$ENV_NAME" python=$PYTHON_VERSION -y
    fi

    # Activate environment
    log_info "Activating environment..."
    eval "$($CONDA_CMD shell.bash hook)"
    $CONDA_CMD activate "$ENV_NAME"

    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    log_info "Using system Python and venv"

    # Check Python version
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PYTHON_VER=$($PYTHON_CMD --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ $(echo "$PYTHON_VER < 3.8" | bc -l) -eq 1 ]]; then
            log_error "Python version $PYTHON_VER is too old. Need >= 3.8"
            exit 1
        fi
        log_warn "Using Python $PYTHON_VER (recommended: 3.11)"
    else
        log_error "Python 3 not found. Please install Python >= 3.8"
        exit 1
    fi

    # Create venv
    VENV_DIR="$PROJECT_DIR/venv"
    if [ -d "$VENV_DIR" ]; then
        log_warn "Virtual environment already exists at $VENV_DIR"
        read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        fi
    fi

    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment at $VENV_DIR"
        $PYTHON_CMD -m venv "$VENV_DIR"
    fi

    # Activate venv
    source "$VENV_DIR/bin/activate"
    PYTHON_CMD="python"
    PIP_CMD="pip"
fi

log_success "Python environment ready: $($PYTHON_CMD --version)"

# Step 3: Upgrade pip
log_info "Upgrading pip..."
$PIP_CMD install --upgrade pip setuptools wheel

# Step 4: Install PyTorch
log_info "Installing PyTorch..."

if [ "$CPU_ONLY" = true ]; then
    log_info "Installing CPU-only PyTorch"
    $PIP_CMD install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    # Install PyTorch with CUDA
    case $CUDA_VERSION in
        11.8)
            TORCH_URL="https://download.pytorch.org/whl/cu118"
            ;;
        12.1)
            TORCH_URL="https://download.pytorch.org/whl/cu121"
            ;;
        12.4)
            TORCH_URL="https://download.pytorch.org/whl/cu124"
            ;;
        *)
            log_error "Unsupported CUDA version: $CUDA_VERSION"
            exit 1
            ;;
    esac

    log_info "Installing PyTorch with CUDA $CUDA_VERSION from $TORCH_URL"
    $PIP_CMD install torch torchvision --index-url "$TORCH_URL"
fi

# Verify PyTorch installation
log_info "Verifying PyTorch installation..."
$PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}')"
if [ "$CPU_ONLY" = false ]; then
    CUDA_AVAILABLE=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        log_success "CUDA is available in PyTorch"
        $PYTHON_CMD -c "import torch; print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
    else
        log_warn "CUDA is NOT available in PyTorch (but was expected)"
        log_warn "Training will be slow. Check your CUDA/driver installation."
    fi
fi

# Step 5: Install project dependencies
log_info "Installing project dependencies..."
$PIP_CMD install -r requirements.txt

# Step 6: Setup custom modules
log_info "Setting up custom modules in ultralytics..."
$PYTHON_CMD setup_env.py

if [ $? -eq 0 ]; then
    log_success "Custom modules installed successfully"
else
    log_error "Failed to setup custom modules"
    exit 1
fi

# Step 7: Verify installation
log_info "Running verification tests..."

VERIFY_SCRIPT=$(cat <<'EOF'
import sys

def verify():
    errors = []

    # Check imports
    try:
        import torch
        import torchvision
        import ultralytics
        from ultralytics.nn.modules.custom import EMA, BiFPN_Concat, SGFA, SOA_SAFM
        print("✓ All modules imported successfully")
    except ImportError as e:
        errors.append(f"Import error: {e}")
        return errors

    # Check versions
    print(f"✓ Python: {sys.version.split()[0]}")
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ Torchvision: {torchvision.__version__}")
    print(f"✓ Ultralytics: {ultralytics.__version__}")

    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ Running in CPU mode")

    # Quick functional test
    try:
        x = torch.randn(1, 256, 32, 32)
        ema = EMA(256)
        y = ema(x)
        assert y.shape == x.shape
        print("✓ Custom modules functional test passed")
    except Exception as e:
        errors.append(f"Functional test failed: {e}")

    return errors

if __name__ == "__main__":
    errors = verify()
    if errors:
        print("\n❌ Verification failed:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("\n✅ All verification tests passed!")
        sys.exit(0)
EOF
)

$PYTHON_CMD -c "$VERIFY_SCRIPT"

if [ $? -eq 0 ]; then
    log_success "Installation completed successfully!"
else
    log_error "Installation verification failed"
    exit 1
fi

# Step 8: Print next steps
echo ""
echo "============================================================"
echo "  Installation Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
if [ "$SKIP_CONDA" = false ]; then
    echo "1. Activate the environment:"
    echo "   conda activate $ENV_NAME"
else
    echo "1. Activate the environment:"
    echo "   source venv/bin/activate"
fi
echo ""
echo "2. Prepare your datasets (see README.md):"
echo "   - Download VisDrone2019, AI-TOD, or NWPU datasets"
echo "   - Run conversion scripts in tools/"
echo ""
echo "3. Start training:"
echo "   python train.py --exp baseline"
echo "   python train.py --exp uav_yolo_pt"
echo ""
echo "For help: python train.py --help"
echo ""
if [ "$CPU_ONLY" = true ]; then
    log_warn "Note: CPU-only mode will be very slow for training."
    log_warn "Consider using a GPU instance for better performance."
fi
echo ""
