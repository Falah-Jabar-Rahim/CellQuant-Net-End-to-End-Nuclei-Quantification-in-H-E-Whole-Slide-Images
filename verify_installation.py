import os
import shutil
import platform
import multiprocessing
import psutil

print("=" * 60)
print("CellQuant-Net Installation Verification")
print("=" * 60)

# System info
print(f"OS: {platform.platform()}")
print(f"CPU cores: {multiprocessing.cpu_count()}")

ram_gb = psutil.virtual_memory().total / (1024**3)
print(f"RAM: {ram_gb:.1f} GB")

disk_gb = shutil.disk_usage("/").free / (1024**3)
print(f"Free disk: {disk_gb:.1f} GB")

# Recommended workers
workers = max(1, multiprocessing.cpu_count() - 2)
print(f"Recommended num_workers: {workers}")

# PyTorch
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")

    if torch.cuda.is_available():
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[OK] CUDA: {torch.version.cuda}")

        props = torch.cuda.get_device_properties(0)
        print(f"[OK] GPU Memory: {props.total_memory/(1024**3):.1f} GB")
    else:
        print("[WARNING] CUDA unavailable")
except Exception as e:
    print(f"[FAIL] PyTorch: {e}")

# TorchVision
try:
    import torchvision
    print(f"[OK] TorchVision {torchvision.__version__}")
except Exception as e:
    print(f"[FAIL] TorchVision: {e}")

# CuPy
try:
    import cupy
    print(f"[OK] CuPy {cupy.__version__}")
except Exception as e:
    print(f"[FAIL] CuPy: {e}")

# Numba
try:
    import numba
    print(f"[OK] Numba {numba.__version__}")
except Exception as e:
    print(f"[FAIL] Numba: {e}")

# OpenSlide
try:
    import openslide
    print("[OK] OpenSlide")
except Exception as e:
    print(f"[FAIL] OpenSlide: {e}")

# PyVIPS
try:
    import pyvips
    print("[OK] PyVIPS")
except Exception as e:
    print(f"[FAIL] PyVIPS: {e}")

# MONAI
try:
    import monai
    print(f"[OK] MONAI {monai.__version__}")
except Exception as e:
    print(f"[FAIL] MONAI: {e}")

# CuCIM
try:
    import cucim
    print(f"[OK] CuCIM {cucim.__version__}")
except Exception as e:
    print(f"[FAIL] CuCIM: {e}")

# CUTLASS extension
try:
    import depthwise_conv2d_implicit_gemm
    print("[OK] depthwise_conv2d_implicit_gemm")
except Exception as e:
    print(f"[FAIL] depthwise_conv2d_implicit_gemm: {e}")

print("=" * 60)
print("Verification completed")
print("=" * 60)
