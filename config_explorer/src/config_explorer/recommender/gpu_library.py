"""GPU library with preloaded specs for common GPUs.

This module provides a library of well-known GPU specifications that users can
select from, with flexibility to override or extend with custom specs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class GPUSpec:
    """Represents GPU hardware specifications.

    Attributes:
        name: Name/model of the GPU
        memory_gb: Total GPU memory in GB
        memory_bandwidth_gb_s: Memory bandwidth in GB/s
        tflops_fp16: Peak FP16 TFLOPS
        tflops_fp32: Peak FP32 TFLOPS
        cost_per_hour: Estimated cost per hour (optional)
    """

    name: str
    memory_gb: float
    memory_bandwidth_gb_s: float
    tflops_fp16: float
    tflops_fp32: float
    cost_per_hour: Optional[float] = None

# Preloaded GPU specifications for common GPUs
# Sources: Official NVIDIA specifications and cloud provider pricing
GPU_LIBRARY: Dict[str, GPUSpec] = {
    "H100": GPUSpec(
        name="NVIDIA H100 80GB",
        memory_gb=80.0,
        memory_bandwidth_gb_s=3350.0,
        tflops_fp16=1979.0,
        tflops_fp32=989.0,
        cost_per_hour=4.76,
    ),
    "H200": GPUSpec(
        name="NVIDIA H200 141GB",
        memory_gb=141.0,
        memory_bandwidth_gb_s=4800.0,
        tflops_fp16=1979.0,
        tflops_fp32=989.0,
        cost_per_hour=5.5,
    ),
    "A100-80GB": GPUSpec(
        name="NVIDIA A100 80GB",
        memory_gb=80.0,
        memory_bandwidth_gb_s=2039.0,
        tflops_fp16=312.0,
        tflops_fp32=156.0,
        cost_per_hour=3.67,
    ),
    "A100-40GB": GPUSpec(
        name="NVIDIA A100 40GB",
        memory_gb=40.0,
        memory_bandwidth_gb_s=1555.0,
        tflops_fp16=312.0,
        tflops_fp32=156.0,
        cost_per_hour=2.93,
    ),
    "L40": GPUSpec(
        name="NVIDIA L40 48GB",
        memory_gb=48.0,
        memory_bandwidth_gb_s=864.0,
        tflops_fp16=362.0,
        tflops_fp32=181.0,
        cost_per_hour=2.24,
    ),
    "L4": GPUSpec(
        name="NVIDIA L4 24GB",
        memory_gb=24.0,
        memory_bandwidth_gb_s=300.0,
        tflops_fp16=120.0,
        tflops_fp32=242.0,
        cost_per_hour=0.796,
    ),
}


def get_gpu_from_library(gpu_key: str) -> Optional[GPUSpec]:
    """Get a GPU spec from the library by key.

    Args:
        gpu_key: Key identifying the GPU (e.g., "H100", "A100-80GB")

    Returns:
        GPUSpec if found, None otherwise
    """
    return GPU_LIBRARY.get(gpu_key)


def list_available_gpus() -> List[str]:
    """List all available GPU keys in the library.

    Returns:
        List of GPU keys
    """
    return list(GPU_LIBRARY.keys())


def get_gpu_specs(gpu_keys: Optional[List[str]] = None) -> List[GPUSpec]:
    """Get GPU specs for specified keys, or all if none specified.

    Args:
        gpu_keys: List of GPU keys to retrieve. If None, returns all GPUs.

    Returns:
        List of GPUSpec objects

    Raises:
        ValueError: If any specified GPU key is not found in the library
    """
    if gpu_keys is None:
        return list(GPU_LIBRARY.values())

    gpus = []
    for key in gpu_keys:
        gpu = get_gpu_from_library(key)
        if gpu is None:
            available = ", ".join(list_available_gpus())
            raise ValueError(
                f"GPU '{key}' not found in library. Available GPUs: {available}"
            )
        gpus.append(gpu)

    return gpus


def create_custom_gpu(
    name: str,
    memory_gb: float,
    memory_bandwidth_gb_s: float,
    tflops_fp16: float,
    tflops_fp32: float,
    cost_per_hour: Optional[float] = None,
) -> GPUSpec:
    """Create a custom GPU spec with user-defined parameters.

    This function allows users to define their own GPU specifications
    that can be used alongside or instead of the preloaded library.

    Args:
        name: Name/model of the GPU
        memory_gb: Total GPU memory in GB
        memory_bandwidth_gb_s: Memory bandwidth in GB/s
        tflops_fp16: Peak FP16 TFLOPS
        tflops_fp32: Peak FP32 TFLOPS
        cost_per_hour: Estimated cost per hour (optional)

    Returns:
        GPUSpec object with the specified parameters
    """
    return GPUSpec(
        name=name,
        memory_gb=memory_gb,
        memory_bandwidth_gb_s=memory_bandwidth_gb_s,
        tflops_fp16=tflops_fp16,
        tflops_fp32=tflops_fp32,
        cost_per_hour=cost_per_hour,
    )