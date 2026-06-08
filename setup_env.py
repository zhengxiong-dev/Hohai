"""
Setup script: registers custom modules (EMA, BiFPN_Concat) into ultralytics.

Run this ONCE after installing ultralytics:
    python setup_env.py

Re-run if you reinstall or upgrade ultralytics.
"""

import importlib
import os
import re
import shutil
import sys


def get_ultralytics_path():
    """Locate the installed ultralytics package."""
    import ultralytics
    return os.path.dirname(ultralytics.__file__)


def patch_tasks_py(ul_path):
    """Patch ultralytics/nn/tasks.py to import and handle custom modules."""
    tasks_path = os.path.join(ul_path, "nn", "tasks.py")

    with open(tasks_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Skip if already patched
    if "# === UAV-YOLO PATCH ===" in content:
        print("[tasks.py] Already patched, skipping.")
        return

    # --- 1. Add import for our custom modules at the top (after existing imports) ---
    import_marker = "from ultralytics.nn.modules import ("
    custom_import = (
        "# === UAV-YOLO PATCH === Custom module imports\n"
        "from ultralytics.nn.modules.custom import EMA, BiFPN_Concat\n"
        "# === END UAV-YOLO PATCH ===\n\n"
    )
    content = content.replace(import_marker, custom_import + import_marker)

    # --- 2. Add EMA to base_modules set ---
    # Find the base_modules definition and add EMA
    content = content.replace(
        "A2C2f,\n        }\n    )\n    repeat_modules",
        "A2C2f,\n            EMA,\n        }\n    )\n    repeat_modules",
    )

    # --- 3. Add BiFPN_Concat handling alongside Concat ---
    content = content.replace(
        "elif m is Concat:",
        "elif m is Concat or m is BiFPN_Concat:",
    )

    with open(tasks_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[tasks.py] Patched successfully: {tasks_path}")


def install_custom_modules(ul_path):
    """Copy custom modules into ultralytics/nn/modules/custom.py."""
    src = os.path.join(os.path.dirname(__file__), "models", "modules.py")
    dst = os.path.join(ul_path, "nn", "modules", "custom.py")

    shutil.copy2(src, dst)
    print(f"[modules] Copied custom.py to: {dst}")

    # Update __init__.py to import custom modules
    init_path = os.path.join(ul_path, "nn", "modules", "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        init_content = f.read()

    if "from .custom import" not in init_content:
        # Add import at the end of the imports section
        init_content += (
            "\n# === UAV-YOLO PATCH ===\n"
            "from .custom import EMA, BiFPN_Concat\n"
            "# === END UAV-YOLO PATCH ===\n"
        )
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(init_content)
        print(f"[__init__.py] Updated: {init_path}")
    else:
        print("[__init__.py] Already patched, skipping.")


def verify_patch():
    """Verify that the patch works correctly."""
    # Force re-import
    mods_to_remove = [k for k in sys.modules if k.startswith("ultralytics")]
    for k in mods_to_remove:
        del sys.modules[k]

    try:
        from ultralytics.nn.modules.custom import EMA, BiFPN_Concat
        print(f"\n[VERIFY] EMA imported successfully: {EMA}")
        print(f"[VERIFY] BiFPN_Concat imported successfully: {BiFPN_Concat}")

        import torch
        # Quick test EMA
        ema = EMA(256)
        x = torch.randn(1, 256, 32, 32)
        y = ema(x)
        assert y.shape == x.shape, f"EMA shape mismatch: {y.shape} vs {x.shape}"
        print(f"[VERIFY] EMA forward pass OK: {x.shape} -> {y.shape}")

        # Quick test BiFPN_Concat
        bfp = BiFPN_Concat(dimension=1)
        a = torch.randn(1, 128, 32, 32)
        b = torch.randn(1, 256, 32, 32)
        c = bfp([a, b])
        assert c.shape == (1, 384, 32, 32), f"BiFPN shape: {c.shape}"
        print(f"[VERIFY] BiFPN_Concat forward pass OK: 128+256 -> {c.shape[1]}")

        print("\n=== All patches applied and verified successfully! ===")
        return True
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  UAV-YOLO Environment Setup")
    print("  Patching ultralytics to support custom modules")
    print("=" * 60)

    ul_path = get_ultralytics_path()
    print(f"\nUltralytics path: {ul_path}\n")

    # Step 1: Copy custom modules
    install_custom_modules(ul_path)

    # Step 2: Patch tasks.py
    patch_tasks_py(ul_path)

    # Step 3: Verify
    print("\nVerifying patches...")
    verify_patch()


if __name__ == "__main__":
    main()
