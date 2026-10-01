"""
check_environment.py

Simple environment report before training.

It only PRINTS information. It does not install or change anything.

Run:
    python scripts/check_environment.py
"""

import os
import platform
import sys

# ----------------------------------------------------------------------------
# 1. Python version
# ----------------------------------------------------------------------------
print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)

print("")
print("[1] PYTHON")
print("    version      :", platform.python_version())
print("    executable   :", sys.executable)
print("    64-bit       :", sys.maxsize > 2 ** 32)

# ----------------------------------------------------------------------------
# 2. PyTorch
# ----------------------------------------------------------------------------
print("")
print("[2] PYTORCH")

try:
    import torch

    print("    version      :", torch.__version__)
    print("    device count :", torch.cuda.device_count())

    cuda_available = torch.cuda.is_available()
    print("    CUDA avail   :", cuda_available)

    if cuda_available:
        print("    CUDA version :", torch.version.cuda)
        for gpu_index in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(gpu_index)
            gpu_memory = torch.cuda.get_device_properties(gpu_index).total_memory
            gpu_memory_gb = round(gpu_memory / (1024 ** 3), 2)
            print("    GPU", gpu_index, ":", gpu_name, "|", gpu_memory_gb, "GB")
    else:
        print("    (no CUDA GPU -> training will run on CPU)")

except ImportError:
    print("    NOT INSTALLED -> run:  python -m pip install torch")
    torch = None
    cuda_available = False

# ----------------------------------------------------------------------------
# 3. Ultralytics
# ----------------------------------------------------------------------------
print("")
print("[3] ULTRALYTICS")

try:
    import ultralytics

    print("    version      :", ultralytics.__version__)

except ImportError:
    print("    NOT INSTALLED -> run:  python -m pip install ultralytics")

# ----------------------------------------------------------------------------
# 4. CPU and RAM (matters because this machine may train on CPU)
# ----------------------------------------------------------------------------
print("")
print("[4] CPU / RAM")
print("    processor    :", platform.processor())
print("    logical CPUs :", os.cpu_count())

try:
    import psutil

    total_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    available_gb = round(psutil.virtual_memory().available / (1024 ** 3), 2)
    print("    RAM total    :", total_gb, "GB")
    print("    RAM available:", available_gb, "GB")

except ImportError:
    print("    RAM          : (psutil not installed, skipping)")

# ----------------------------------------------------------------------------
# 5. Pretrained model file present?
# ----------------------------------------------------------------------------
print("")
print("[5] PRETRAINED WEIGHTS")

model_name = "yolo26n.pt"
model_found = os.path.exists(model_name)
print("    looking for  :", model_name)
print("    in cwd?      :", model_found)

# ----------------------------------------------------------------------------
# 6. Plain-English summary
# ----------------------------------------------------------------------------
print("")
print("=" * 60)
print("SUMMARY")
print("=" * 60)

if torch is not None and cuda_available:
    print("Training device : GPU")
    print("Expect training to be reasonably fast.")
else:
    print("Training device : CPU")
    print("WARNING: CPU training on this dataset will take a long time.")
    print("         Few epochs only. Consider a cloud GPU for the full run.")

print("=" * 60)
