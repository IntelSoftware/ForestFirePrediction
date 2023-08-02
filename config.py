import warnings

# ignore warnings from packages, be careful with this
warnings.filterwarnings("ignore")

import os
import random
import torch

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    print("Failed to import intel_extension_for_pytorch.")
    ipex = None

try:
    import psutil
except ImportError:
    print("Failed to import psutil.")
    psutil = None

import fnmatch
import pathlib
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

# Check XPU availability
HAS_XPU = torch.xpu.is_available() if ipex else False


def set_config(device: torch.device):
    """Set CPU and XPU configuration for torch."""
    if device == torch.device("xpu"):
        # pvc 1550 has 2 tiles, ipex would use it as one single device
        os.environ["IPEX_TILE_AS_DEVICE"] = "0"
    if psutil:
        num_physical_cores = psutil.cpu_count(logical=False)
        os.environ["OMP_NUM_THREADS"] = str(num_physical_cores)
        print(f"OMP_NUM_THREADS set to: {num_physical_cores}")
    else:
        print("psutil not found. Unable to set OMP_NUM_THREADS.")


def set_seed(seed_value: int = 42):
    """Set all random seeds using `seed_value`."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if HAS_XPU:
        torch.xpu.manual_seed(seed_value)
        torch.xpu.manual_seed_all(seed_value)
        torch.backends.mkldnn.deterministic = True
        torch.backends.mkldnn.benchmark = False


def set_device(device=None):
    """
    Sets the device for PyTorch. If a specific device is specified, it will be used.
    Otherwise, it will default to CPU or XPU based on availability.
    """
    if device is not None:
        print(f"Device set to {device} by user.")
        return torch.device(device)

    if HAS_XPU:
        device_count = torch.xpu.device_count()
        device_id = random.randint(0, int(device_count) - 1)
        device = f"xpu:{device_id}"
        print(f"XPU devices detected, using {device}")
        print(f"XPU device name: {torch.xpu.get_device_name(0)}")
        return torch.device(device)

    return torch.device("cpu")


def seed_everything(seed: int = 4242):
    """Set all random seeds using `seed`."""
    print(f"seed set to: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed)


def ncores() -> int:
    """Get number of physical cores."""
    if psutil:
        return psutil.cpu_count(logical=False)
    else:
        print("psutil not found. Unable to count cores.")
        return None


# Configuration
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "1"
device = set_device("xpu")
set_config(device)
