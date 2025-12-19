"""
GPU Recommender: Recommend GPUs for LLM inference based on model size and requirements.

This module provides a high-level API for recommending GPU configurations for
serving LLM models. It integrates:
- Model analysis using HuggingFace transformers and capacity_planner
- GPU library with specifications for common GPUs
- Tensor parallelism (TP) calculation to find feasible configurations
- Performance estimation using llm-optimizer
- KVCacheDetail from capacity_planner for detailed KV cache analysis

Example usage:
    from config_explorer.recommender import recommend_gpus, ModelArchitecture

    # Load model
    model = ModelArchitecture("meta-llama/Llama-3.1-8B")

    # Get recommendations
    recs = recommend_gpus(
        model=model,
        input_length=1024,
        output_length=512,
        precision="fp16"
    )

    # Display results
    print(recs.summary())

    # Filter to fitting configurations only
    fitting = recs.filter_fitting_only()

    # Sort by performance
    fitting.sort_by_performance()

    # Access KV cache details
    kv_detail = model.get_kv_cache_detail(context_length=4096)
    print(f"Attention type: {kv_detail.attention_type}")
    print(f"Per-token memory: {kv_detail.per_token_memory_bytes} bytes")
"""

from .model_architecture import ModelArchitecture
from .gpu_library import GPUSpec, GPU_LIBRARY, get_gpu_specs, list_available_gpus
from .recommendation import (
    GPURecommendation,
    find_gpu_recommendations,
    filter_recommendations_by_budget,
)
from .performance import (
    PerformanceEstimate,
    estimate_performance,
    estimate_performance_batch,
)
from .results import (
    RecommendationWithPerformance,
    RecommendationResult,
)
from ..capacity_planner import KVCacheDetail


def recommend_gpus(
    model: ModelArchitecture,
    input_length: int = 1024,
    output_length: int = 512,
    precision: str = None,
    context_length: int = 2048,
    batch_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_tensor_parallel: int = None,
    estimate_performance_flag: bool = True,
    max_gpus: int = None,
) -> RecommendationResult:
    """
    Recommend GPUs for serving an LLM model.

    This function:
    1. Finds viable GPU + tensor parallelism (TP) combinations that fit the model
    2. Calculates available memory for KV cache using KVCacheDetail
    3. Estimates inference performance (TTFT, ITL, throughput)
    4. Returns results in structured, human-readable format

    Extensively reuses capacity_planner's KVCacheDetail for accurate KV cache
    calculations including attention mechanism detection and per-token memory.
    
    Args:
        model: ModelArchitecture instance
        input_length: Expected input prompt length (tokens)
        output_length: Expected output generation length (tokens)
        precision: Model precision to use (e.g., 'fp32', 'fp16', 'int4').
                   If None, automatically inferred from model config using capacity_planner.
        context_length: Maximum context length for KV cache calculations
        batch_size: Batch size for KV cache calculations
        gpu_memory_utilization: GPU memory utilization factor (0.0-1.0)
        max_tensor_parallel: Maximum tensor parallelism to consider
        estimate_performance_flag: Whether to estimate performance using llm-optimizer
        max_gpus: Maximum number of GPUs to recommend
    """
    
    # Infer precision from model if not specified
    if precision is None:
        precision = model.inferred_precision_name

    # Find viable GPU/TP combinations
    recommendations = find_gpu_recommendations(
        model=model,
        context_length=context_length,
        batch_size=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_tensor_parallel=max_tensor_parallel,
        include_all_gpus=False,
    )

    # Filter by GPU budget if specified
    if max_gpus is not None:
        recommendations = filter_recommendations_by_budget(
            recommendations,
            max_gpus=max_gpus,
        )

    # Create result object
    result = RecommendationResult(
        model_id=model.model_id,
        input_length=input_length,
        output_length=output_length,
        precision=precision,
    )

    # Add recommendations
    for rec in recommendations:
        # Optionally estimate performance
        perf = None
        if estimate_performance_flag:
            perf = estimate_performance(
                model_id=model.model_id,
                gpu_name=rec.gpu_spec.name,
                num_gpus=rec.total_gpus,
                tensor_parallel_size=rec.tensor_parallel_size,
                input_length=input_length,
                output_length=output_length,
                batch_size=batch_size,
            )

        result.add_recommendation(rec, perf)

    return result


__all__ = [
    # Main API
    "recommend_gpus",

    # Model
    "ModelArchitecture",

    # GPU Library
    "GPUSpec",
    "GPU_LIBRARY",
    "get_gpu_spec",
    "list_available_gpus",

    # Recommendations
    "GPURecommendation",
    "find_gpu_recommendations",
    "filter_recommendations_by_budget",

    # Performance
    "PerformanceEstimate",
    "estimate_performance",
    "estimate_performance_batch",

    # Results
    "RecommendationWithPerformance",
    "RecommendationResult",
]
