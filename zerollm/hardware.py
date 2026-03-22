"""Auto-detect hardware: CPU, GPU, RAM, and recommend optimal settings."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from functools import lru_cache

import psutil


@dataclass
class HardwareInfo:
    """Detected hardware profile."""

    platform: str  # "darwin", "linux", "windows"
    arch: str  # "arm64", "x86_64"
    cpu: str  # "Apple M2", "Intel i7-12700", etc.
    ram_gb: float
    gpu_type: str | None  # "metal", "cuda", "rocm", None
    gpu_name: str | None  # "Apple M2", "NVIDIA RTX 4090", etc.
    gpu_vram_gb: float | None  # None for unified memory
    n_threads: int  # total available threads
    recommended_n_gpu_layers: int = -1  # -1 = all layers on GPU
    recommended_threads: int = 4

    @property
    def has_gpu(self) -> bool:
        return self.gpu_type is not None

    def summary(self) -> str:
        """One-line summary for display."""
        parts = [self.cpu, f"{self.ram_gb:.0f}GB RAM"]
        if self.gpu_type:
            parts.append(f"{self.gpu_type.upper()} GPU")
            if self.gpu_name and self.gpu_name != self.cpu:
                parts.append(f"({self.gpu_name})")
        else:
            parts.append("CPU only")
        return ", ".join(parts)


def _get_cpu_name() -> str:
    """Get a human-readable CPU name."""
    system = platform.system()

    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        # Apple Silicon doesn't have brand_string, use chip name
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.chip"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":")[1].strip()
        except FileNotFoundError:
            pass

    elif system == "Windows":
        return platform.processor() or "Unknown CPU"

    return platform.processor() or "Unknown CPU"


def _detect_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _detect_cuda() -> tuple[bool, str | None, float | None]:
    """Check for NVIDIA CUDA GPU. Returns (available, name, vram_gb)."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return True, name, round(vram, 1)
    except ImportError:
        pass

    # Fallback: try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            name = parts[0].strip()
            vram = float(parts[1].strip()) / 1024  # MB to GB
            return True, name, round(vram, 1)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    return False, None, None


def _detect_rocm() -> tuple[bool, str | None]:
    """Check for AMD ROCm GPU."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "GPU" in line or "Card" in line:
                    return True, line.strip()
            return True, "AMD GPU"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False, None


@lru_cache(maxsize=1)
def detect() -> HardwareInfo:
    """Detect hardware and return optimal settings. Result is cached."""
    system = platform.system().lower()
    if system == "windows":
        system = "windows"
    arch = platform.machine()
    cpu_name = _get_cpu_name()
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    n_threads = os.cpu_count() or 4

    # Detect GPU
    gpu_type = None
    gpu_name = None
    gpu_vram_gb = None

    if _detect_apple_silicon():
        gpu_type = "metal"
        gpu_name = cpu_name  # unified architecture
        gpu_vram_gb = ram_gb  # shared memory
    else:
        cuda_available, cuda_name, cuda_vram = _detect_cuda()
        if cuda_available:
            gpu_type = "cuda"
            gpu_name = cuda_name
            gpu_vram_gb = cuda_vram
        else:
            rocm_available, rocm_name = _detect_rocm()
            if rocm_available:
                gpu_type = "rocm"
                gpu_name = rocm_name

    # Recommended settings
    recommended_threads = max(1, n_threads - 2)  # leave 2 threads for system
    recommended_n_gpu_layers = -1 if gpu_type else 0

    return HardwareInfo(
        platform=system,
        arch=arch,
        cpu=cpu_name,
        ram_gb=ram_gb,
        gpu_type=gpu_type,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        n_threads=n_threads,
        recommended_n_gpu_layers=recommended_n_gpu_layers,
        recommended_threads=recommended_threads,
    )


def compute_n_gpu_layers(total_layers: int, power: float) -> int:
    """Convert power (0.0-1.0) to number of GPU layers."""
    if power <= 0.0:
        return 0
    if power >= 1.0:
        return -1  # all layers
    return max(1, int(total_layers * power))


def compute_threads(power: float, hw: HardwareInfo | None = None) -> int:
    """Convert power (0.0-1.0) to number of CPU threads."""
    if hw is None:
        hw = detect()
    max_threads = hw.recommended_threads
    return max(1, int(max_threads * max(power, 0.1)))
