"""
GPU Recommendation: Core logic for recommending GPUs for LLM inference.

Finds viable GPU/TP combinations based on model size and available GPU memory.
Extensively reuses KVCacheDetail from capacity_planner for accurate memory calculations.
"""

from dataclasses import dataclass
from typing import Optional
import math

from .model_architecture import ModelArchitecture
from .gpu_library import GPUSpec, GPU_LIBRARY


@dataclass
class GPURecommendation:
    """
    A single GPU recommendation: GPU + tensor parallelism configuration.
    """

    gpu_spec: GPUSpec
    """GPU specification."""

    tensor_parallel_size: int
    """Number of GPUs for tensor parallelism."""

    total_gpus: int
    """Total GPUs needed (TP + DP/replication)."""

    model_fits: bool
    """Whether the model fits on this GPU with this TP configuration."""

    memory_per_gpu_gb: float
    """Model memory required per GPU in GB."""

    available_memory_for_kv_gb: float
    """Available memory per GPU for KV cache in GB."""

    max_concurrent_requests: int
    """Maximum concurrent requests this config can handle (at default context length)."""

    max_batch_size: int
    """Maximum batch size on this configuration."""

    def fits_memory(self) -> bool:
        """Check if model fits in memory."""
        return self.model_fits

    def __repr__(self) -> str:
        status = "✓" if self.model_fits else "✗"
        return (
            f"{status} {self.gpu_spec.name}x{self.tensor_parallel_size} "
            f"({self.memory_per_gpu_gb:.1f}GB/GPU)"
        )


def find_gpu_recommendations(
    model: ModelArchitecture,
    context_length: int = 2048,
    batch_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_tensor_parallel: Optional[int] = None,
    include_all_gpus: bool = False,
) -> list[GPURecommendation]:
    """
    Find viable GPU and tensor parallelism combinations for a model.

    Algorithm:
    1. For each GPU in library:
       - Calculate memory per GPU needed for model
       - If TP=1 and model fits: add recommendation
       - If model doesn't fit with TP=1: try TP=2,4,8,16... until it fits
       - Skip if TP exceeds max_tensor_parallel or possible TP values

    2. For each viable config, calculate KV cache memory available using KVCacheDetail
    
    This function extensively reuses capacity_planner's KVCacheDetail for:
    - Accurate per-token KV cache memory calculation
    - Attention mechanism detection (MLA, MHA, GQA, MQA)
    - Per-request and batch KV cache size calculation
    """

    recommendations = []
    model_weights_memory = model.model_memory_gb()
    possible_tp_values = model.possible_tensor_parallel_sizes()

    # Get KVCacheDetail to understand model's KV cache characteristics
    kv_detail = model.get_kv_cache_detail(context_length, batch_size=1)

    # Determine max TP to consider
    if max_tensor_parallel is None:
        # Use the largest possible TP
        max_tensor_parallel = max(possible_tp_values) if possible_tp_values else 1

    for gpu_name, gpu_spec in sorted(GPU_LIBRARY.items()):
        # Try different TP values
        tps_to_try = [tp for tp in possible_tp_values if tp <= max_tensor_parallel]
        if not tps_to_try or len(tps_to_try) == 0:
            tps_to_try = [1]

        for tp in tps_to_try:
            # Memory per GPU with this TP
            memory_per_gpu = model_weights_memory / tp

            # Check if model fits
            model_fits = memory_per_gpu <= gpu_spec.memory_gb * gpu_memory_utilization

            # Skip if model doesn't fit and we're not including all
            if not model_fits and not include_all_gpus:
                continue

            # Calculate available KV cache memory per GPU
            available_memory = gpu_spec.memory_gb * gpu_memory_utilization - memory_per_gpu
            available_memory = max(0, available_memory)

            # Calculate maximum concurrent requests using per-request KV cache from KVCacheDetail
            # KVCacheDetail.per_request_kv_cache_gb is the memory for a single request with the context length
            per_request_kv = kv_detail.per_request_kv_cache_gb
            max_concurrent = 0
            if per_request_kv > 0:
                max_concurrent = int(available_memory / per_request_kv)

            # Max batch size (assuming single request context)
            max_batch = 0
            if per_request_kv > 0:
                max_batch = int(available_memory / per_request_kv) + 1

            rec = GPURecommendation(
                gpu_spec=gpu_spec,
                tensor_parallel_size=tp,
                total_gpus=tp,
                model_fits=model_fits,
                memory_per_gpu_gb=memory_per_gpu,
                available_memory_for_kv_gb=available_memory,
                max_concurrent_requests=max_concurrent,
                max_batch_size=max_batch,
            )

            recommendations.append(rec)

    # Sort by GPU memory (ascending)
    recommendations.sort(key=lambda r: r.gpu_spec.memory_gb)

    return recommendations


def filter_recommendations_by_budget(
    recommendations: list[GPURecommendation],
    max_gpus: int,
    min_memory_per_gpu_gb: Optional[float] = None,
) -> list[GPURecommendation]:
    """
    Filter recommendations based on budget constraints.

    Args:
        recommendations: List of GPURecommendation objects
        max_gpus: Maximum number of GPUs to use
        min_memory_per_gpu_gb: Minimum memory per GPU (None = no constraint)

    Returns:
        Filtered list of recommendations
    """
    filtered = [r for r in recommendations if r.total_gpus <= max_gpus]

    if min_memory_per_gpu_gb is not None:
        filtered = [r for r in filtered if r.gpu_spec.memory_gb >= min_memory_per_gpu_gb]

    return filtered
