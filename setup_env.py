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

    # Check if already patched and update if needed
    if "# === UAV-YOLO PATCH ===" in content:
        # Update the import line if SOA_SAFM is missing
        old_import = "from ultralytics.nn.modules.custom import EMA, BiFPN_Concat, SGFA"
        new_import = "from ultralytics.nn.modules.custom import EMA, BiFPN_Concat, SGFA, SOA_SAFM"
        if old_import in content and new_import not in content:
            content = content.replace(old_import, new_import)
            print("[tasks.py] Updating existing patch to include SOA_SAFM...")
        else:
            print("[tasks.py] Already up to date, skipping.")
            with open(tasks_path, "w", encoding="utf-8") as f:
                f.write(content)
            return

    # --- 1. Add import for our custom modules at the top (after existing imports) ---
    import_marker = "from ultralytics.nn.modules import ("
    custom_import = (
        "# === UAV-YOLO PATCH === Custom module imports\n"
        "from ultralytics.nn.modules.custom import EMA, BiFPN_Concat, SGFA, SOA_SAFM\n"
        "# === END UAV-YOLO PATCH ===\n\n"
    )
    content = content.replace(import_marker, custom_import + import_marker)

    # --- 2. Add EMA to base_modules set ---
    # Find the base_modules definition and add EMA
    content = content.replace(
        "A2C2f,\n        }\n    )\n    repeat_modules",
        "A2C2f,\n            EMA,\n        }\n    )\n    repeat_modules",
    )

    # --- 3. Add BiFPN_Concat and SGFA handling alongside Concat ---
    content = content.replace(
        "elif m is Concat:",
        "elif m is Concat or m is BiFPN_Concat or m is SGFA:",
    )

    # --- 4. Add SOA_SAFM special handling (output channels = target input channels) ---
    # Insert after the Concat handling
    safm_handling = (
        "        elif m is SOA_SAFM:\n"
        "            target_index = args[0] if args else 0\n"
        "            c2 = ch[f[target_index]]\n"
    )
    # Find the line after "elif m is Concat..." block and insert SOA_SAFM handling
    concat_block = "elif m is Concat or m is BiFPN_Concat or m is SGFA:\n            c2 = sum(ch[x] for x in f)"
    if concat_block in content and safm_handling.strip() not in content:
        content = content.replace(
            concat_block,
            concat_block + "\n" + safm_handling
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
            "from .custom import EMA, BiFPN_Concat, SGFA, SOA_SAFM\n"
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
        from ultralytics.nn.modules.custom import EMA, BiFPN_Concat, SGFA, SOA_SAFM
        print(f"\n[VERIFY] EMA imported successfully: {EMA}")
        print(f"[VERIFY] BiFPN_Concat imported successfully: {BiFPN_Concat}")
        print(f"[VERIFY] SGFA imported successfully: {SGFA}")
        print(f"[VERIFY] SOA_SAFM imported successfully: {SOA_SAFM}")

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

        # Quick test SGFA
        sgfa = SGFA(dimension=1, ratio=1.0)
        f_high = torch.randn(1, 512, 16, 16)
        f_low = torch.randn(1, 256, 32, 32)
        out = sgfa([f_high, f_low])
        # Output should be concat of projected features
        print(f"[VERIFY] SGFA forward pass OK: {f_high.shape} + {f_low.shape} -> {out.shape}")
        assert out.shape[2:] == f_low.shape[2:], f"SGFA spatial shape mismatch"
        assert out.shape[1] == 256 + 512, f"SGFA channel mismatch: {out.shape[1]} vs 768"

        # Quick test SOA_SAFM
        safm = SOA_SAFM(target_index=0, ratio=1.0)
        p2 = torch.randn(1, 128, 32, 32)
        p3 = torch.randn(1, 256, 16, 16)
        p4 = torch.randn(1, 512, 8, 8)
        out = safm([p2, p3, p4])
        # Output should have target scale spatial size and channels
        print(f"[VERIFY] SOA_SAFM forward pass OK: [P2, P3, P4] -> {out.shape}")
        assert out.shape[2:] == p2.shape[2:], f"SOA_SAFM spatial mismatch"
        assert out.shape[1] == 128, f"SOA_SAFM channel mismatch: {out.shape[1]} vs 128"

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
