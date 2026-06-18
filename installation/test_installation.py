#!/usr/bin/env python3
"""
Installation Verification Script
Quick test to verify UAV-YOLO environment is set up correctly
"""

import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")

    tests = {
        "Python": sys.version_info >= (3, 8),
        "torch": False,
        "torchvision": False,
        "ultralytics": False,
        "pycocotools": False,
        "cv2 (opencv)": False,
        "numpy": False,
        "pandas": False,
        "matplotlib": False,
    }

    # Test torch
    try:
        import torch
        tests["torch"] = True
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError:
        print("  ✗ PyTorch not found")

    # Test torchvision
    try:
        import torchvision
        tests["torchvision"] = True
        print(f"  ✓ Torchvision {torchvision.__version__}")
    except ImportError:
        print("  ✗ Torchvision not found")

    # Test ultralytics
    try:
        import ultralytics
        tests["ultralytics"] = True
        print(f"  ✓ Ultralytics {ultralytics.__version__}")
    except ImportError:
        print("  ✗ Ultralytics not found")

    # Test pycocotools
    try:
        import pycocotools
        tests["pycocotools"] = True
        print(f"  ✓ pycocotools")
    except ImportError:
        print("  ✗ pycocotools not found")

    # Test opencv
    try:
        import cv2
        tests["cv2 (opencv)"] = True
        print(f"  ✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("  ✗ OpenCV not found")

    # Test numpy
    try:
        import numpy as np
        tests["numpy"] = True
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError:
        print("  ✗ NumPy not found")

    # Test pandas
    try:
        import pandas as pd
        tests["pandas"] = True
        print(f"  ✓ Pandas {pd.__version__}")
    except ImportError:
        print("  ✗ Pandas not found")

    # Test matplotlib
    try:
        import matplotlib
        tests["matplotlib"] = True
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("  ✗ Matplotlib not found")

    return all(tests.values())


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.version.cuda}")
            print(f"  ✓ Device count: {torch.cuda.device_count()}")
            print(f"  ✓ Device name: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("  ⚠ CUDA not available (CPU mode)")
            return True  # Not an error, just CPU mode
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def test_custom_modules():
    """Test custom UAV-YOLO modules"""
    print("\nTesting custom modules...")

    try:
        from ultralytics.nn.modules.custom import EMA, BiFPN_Concat, SGFA, SOA_SAFM
        print("  ✓ EMA imported")
        print("  ✓ BiFPN_Concat imported")
        print("  ✓ SGFA imported")
        print("  ✓ SOA_SAFM imported")

        # Functional test
        import torch
        x = torch.randn(1, 256, 32, 32)
        ema = EMA(256)
        y = ema(x)

        if y.shape == x.shape:
            print("  ✓ Functional test passed")
            return True
        else:
            print(f"  ✗ Shape mismatch: {y.shape} vs {x.shape}")
            return False
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        print("\n  💡 Tip: Run 'python setup_env.py' to install custom modules")
        return False
    except Exception as e:
        print(f"  ✗ Functional test failed: {e}")
        return False


def test_model_loading():
    """Test if YOLO model can be loaded"""
    print("\nTesting model loading...")

    try:
        from ultralytics import YOLO

        # Try to load YOLO11n (smallest model)
        print("  Loading YOLO11n model...")
        model = YOLO("yolo11n.pt")
        print("  ✓ Model loaded successfully")
        return True
    except Exception as e:
        print(f"  ⚠ Model loading test skipped: {e}")
        print("    (This is normal if you haven't trained any models yet)")
        return True  # Not a critical error


def main():
    print("=" * 60)
    print("  UAV-YOLO Installation Verification")
    print("=" * 60)
    print()

    results = {
        "Imports": test_imports(),
        "CUDA": test_cuda(),
        "Custom Modules": test_custom_modules(),
        "Model Loading": test_model_loading(),
    }

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    all_passed = all(results.values())

    print()
    if all_passed:
        print("  🎉 All tests passed! Your environment is ready.")
        print()
        print("  Next steps:")
        print("    1. Prepare datasets (see README.md)")
        print("    2. Run: python train.py --exp baseline")
        return 0
    else:
        print("  ❌ Some tests failed. Please check the errors above.")
        print()
        print("  Common fixes:")
        print("    • Run: python setup_env.py")
        print("    • Run: bash install.sh")
        print("    • Check: pip list | grep torch")
        return 1


if __name__ == "__main__":
    sys.exit(main())
