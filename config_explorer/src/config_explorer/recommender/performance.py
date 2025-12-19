"""
LLM Optimizer Integration: Estimates inference performance using llm-optimizer.

Integrates with bentoml/llm-optimizer to estimate TTFT, ITL, and throughput
for different GPU/TP configurations.
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceEstimate:
    """Performance estimation for a GPU configuration."""

    time_to_first_token_ms: Optional[float] = None
    """Time to first token (TTFT) in milliseconds."""

    inter_token_latency_ms: Optional[float] = None
    """Inter-token latency (ITL) in milliseconds."""

    throughput_tokens_per_sec: Optional[float] = None
    """Output throughput in tokens/second."""

    max_batch_size: Optional[int] = None
    """Maximum batch size for this config."""

    max_concurrent_requests: Optional[int] = None
    """Maximum concurrent requests."""

    error: Optional[str] = None
    """Error message if estimation failed."""

    def is_valid(self) -> bool:
        """Check if estimate is valid (no errors)."""
        return self.error is None

    def __repr__(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        parts = []
        if self.time_to_first_token_ms is not None:
            parts.append(f"TTFT: {self.time_to_first_token_ms:.1f}ms")
        if self.inter_token_latency_ms is not None:
            parts.append(f"ITL: {self.inter_token_latency_ms:.2f}ms")
        if self.throughput_tokens_per_sec is not None:
            parts.append(f"Throughput: {self.throughput_tokens_per_sec:.1f} tok/s")
        return " | ".join(parts) if parts else "No performance data"


def estimate_performance(
    model_id: str,
    gpu_name: str,
    num_gpus: int = 1,
    tensor_parallel_size: int = 1,
    input_length: int = 1024,
    output_length: int = 512,
    batch_size: int = 1,
) -> PerformanceEstimate:
    """
    Estimate inference performance using bentoML's llm-optimizer.

    Uses bentoml/llm-optimizer's estimate_llm_performance function to estimate:
    - Time to first token (TTFT)
    - Inter-token latency (ITL)
    - Output throughput

    Args:
        model_id: HuggingFace model ID (e.g., 'meta-llama/Llama-3.1-70B')
        gpu_name: GPU name (e.g., 'H100', 'A100', 'L40S')
        num_gpus: Total number of GPUs to use
        tensor_parallel_size: Tensor parallelism degree
        input_length: Input prompt length in tokens
        output_length: Output generation length in tokens
        batch_size: Batch size

    Returns:
        PerformanceEstimate object with TTFT, ITL, and throughput metrics

    Note:
        Requires bentoML's llm-optimizer package:
        pip install llm-optimizer
    """

    try:
        # Import here to avoid hard dependency
        from llm_optimizer.performance import estimate_llm_performance

        # Call bentoML's llm-optimizer estimate function
        # Maps our parameters to llm-optimizer's expected format
        result = estimate_llm_performance(
            model_id=model_id,
            gpu_type=gpu_name,
            num_gpu=num_gpus,
            tp_degree=tensor_parallel_size,
            input_len=input_length,
            output_len=output_length,
            batch_size=batch_size,
        )

        # Parse results into PerformanceEstimate
        # llm-optimizer returns a dict or object with metrics
        if isinstance(result, dict):
            estimate = PerformanceEstimate(
                time_to_first_token_ms=result.get("ttft_ms") or result.get("ttft"),
                inter_token_latency_ms=result.get("itl_ms") or result.get("itl"),
                throughput_tokens_per_sec=result.get("throughput") or result.get("tps"),
                max_batch_size=result.get("max_batch_size", batch_size),
                max_concurrent_requests=result.get("max_concurrent_requests"),
            )
        else:
            # Handle object response from llm-optimizer
            estimate = PerformanceEstimate(
                time_to_first_token_ms=getattr(result, "ttft_ms", None) or getattr(result, "ttft", None),
                inter_token_latency_ms=getattr(result, "itl_ms", None) or getattr(result, "itl", None),
                throughput_tokens_per_sec=getattr(result, "throughput", None) or getattr(result, "tps", None),
                max_batch_size=getattr(result, "max_batch_size", batch_size),
                max_concurrent_requests=getattr(result, "max_concurrent_requests", None),
            )
        
        return estimate

    except ImportError:
        return PerformanceEstimate(
            error="llm-optimizer not installed. Install with: pip install bentoml-llm-optimizer"
        )
    except Exception as e:
        logger.warning(f"Failed to estimate performance for {model_id} on {gpu_name}x{num_gpus}: {e}")
        return PerformanceEstimate(
            error=f"Estimation failed: {str(e)}"
        )


def estimate_performance_batch(
    model_id: str,
    gpu_configs: list[tuple[str, int]],  # [(gpu_name, num_gpus), ...]
    input_length: int = 1024,
    output_length: int = 512,
    tensor_parallel_size: int = 1,
) -> dict[str, PerformanceEstimate]:
    """
    Estimate performance for multiple GPU configurations using bentoML's llm-optimizer.

    Efficiently estimates inference performance across different GPU setups,
    useful for finding optimal configurations for a given workload.

    Args:
        model_id: HuggingFace model ID (e.g., 'meta-llama/Llama-3.1-70B')
        gpu_configs: List of (gpu_name, num_gpus) tuples
                    e.g., [('H100', 1), ('H100', 2), ('A100', 4)]
        input_length: Input prompt length in tokens (default: 1024)
        output_length: Output generation length in tokens (default: 512)
        tensor_parallel_size: Tensor parallelism degree (default: 1)

    Returns:
        Dictionary mapping config_name -> PerformanceEstimate
        Example: {
            'H100x1': PerformanceEstimate(...),
            'H100x2': PerformanceEstimate(...),
            'A100x4': PerformanceEstimate(...)
        }

    Example:
        >>> configs = [('H100', 1), ('H100', 2), ('A100', 4)]
        >>> results = estimate_performance_batch(
        ...     'meta-llama/Llama-3.1-70B',
        ...     configs,
        ...     input_length=2048,
        ...     output_length=512
        ... )
        >>> for config, estimate in results.items():
        ...     if estimate.is_valid():
        ...         print(f"{config}: TTFT={estimate.time_to_first_token_ms:.1f}ms")
    """
    results = {}

    for gpu_name, num_gpus in gpu_configs:
        config_name = f"{gpu_name}x{num_gpus}"
        estimate = estimate_performance(
            model_id=model_id,
            gpu_name=gpu_name,
            num_gpus=num_gpus,
            tensor_parallel_size=tensor_parallel_size,
            input_length=input_length,
            output_length=output_length,
        )
        results[config_name] = estimate

    return results
