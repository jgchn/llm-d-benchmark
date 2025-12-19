"""
Performance Estimation Module Documentation

This module integrates bentoML's llm-optimizer for accurate inference performance
estimation across different GPU configurations and tensor parallelism settings.
"""

# ==============================================================================
# OVERVIEW
# ==============================================================================

The `performance.py` module provides two main functions for performance estimation:

1. estimate_performance(): Single configuration estimation
2. estimate_performance_batch(): Multiple configuration estimation


# ==============================================================================
# BENTOML LLM-OPTIMIZER INTEGRATION
# ==============================================================================

The module uses bentoml/llm-optimizer's estimate_llm_performance() function, which
estimates:

  - Time to First Token (TTFT): Latency before first output token
  - Inter-Token Latency (ITL): Time between output tokens  
  - Throughput: Output tokens per second

Parameters Mapped to llm-optimizer:
  model_id       -> model_id          (HuggingFace model ID)
  gpu_name       -> gpu_type          (GPU name: H100, A100, etc.)
  num_gpus       -> num_gpu           (Number of GPUs)
  tensor_parallel_size -> tp_degree   (Tensor parallelism degree)
  input_length   -> input_len         (Prompt length in tokens)
  output_length  -> output_len        (Generation length in tokens)
  batch_size     -> batch_size        (Batch size)

Returns:
  PerformanceEstimate with:
    - time_to_first_token_ms (float, ms)
    - inter_token_latency_ms (float, ms)
    - throughput_tokens_per_sec (float, tok/s)
    - max_batch_size (int)
    - max_concurrent_requests (int)
    - error (str, if estimation failed)


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

Example 1: Single Configuration Estimation
-------------------------------------------

from config_explorer.recommender.performance import estimate_performance

estimate = estimate_performance(
    model_id="meta-llama/Llama-3.1-70B",
    gpu_name="H100",
    num_gpus=1,
    tensor_parallel_size=1,
    input_length=2048,
    output_length=512,
)

if estimate.is_valid():
    print(f"TTFT: {estimate.time_to_first_token_ms:.1f}ms")
    print(f"ITL: {estimate.inter_token_latency_ms:.2f}ms")
    print(f"Throughput: {estimate.throughput_tokens_per_sec:.1f} tok/s")
else:
    print(f"Error: {estimate.error}")


Example 2: Multiple Configuration Batch Estimation
---------------------------------------------------

from config_explorer.recommender.performance import estimate_performance_batch

configs = [
    ("H100", 1),
    ("H100", 2),
    ("A100", 4),
    ("L40S", 8),
]

results = estimate_performance_batch(
    model_id="meta-llama/Llama-3.1-70B",
    gpu_configs=configs,
    input_length=2048,
    output_length=512,
    tensor_parallel_size=1,
)

for config_name, estimate in results.items():
    if estimate.is_valid():
        print(f"{config_name}: TTFT={estimate.time_to_first_token_ms:.1f}ms, "
              f"Throughput={estimate.throughput_tokens_per_sec:.1f} tok/s")


Example 3: Integration with GPU Recommender
--------------------------------------------

from config_explorer.recommender import recommend_gpus, ModelArchitecture

model = ModelArchitecture("meta-llama/Llama-3.1-70B")

# recommend_gpus internally uses estimate_performance for each config
result = recommend_gpus(
    model=model,
    input_length=2048,
    output_length=512,
    estimate_performance_flag=True,  # Enable performance estimation
)

# Results include performance metrics
print(result.summary())  # Shows TTFT, ITL, throughput for each config


# ==============================================================================
# ERROR HANDLING
# ==============================================================================

The module gracefully handles missing llm-optimizer:

1. If llm-optimizer is not installed:
   Returns PerformanceEstimate with error message directing to install

2. If estimation fails for a specific config:
   Returns PerformanceEstimate with error details
   (doesn't crash, allows other configs to be estimated)

3. Invalid model IDs or GPU names:
   Handled by llm-optimizer, returns error in PerformanceEstimate


# ==============================================================================
# INSTALLATION REQUIREMENTS
# ==============================================================================

Required:
  pip install bentoml-llm-optimizer
  # or
  pip install git+https://github.com/bentoml/llm-optimizer.git

The module uses dynamic import to make it optional (recommender works without it,
but performance estimation is skipped).


# ==============================================================================
# PERFORMANCE NOTES
# ==============================================================================

1. First call may be slow (model analysis and caching)
2. Subsequent calls are faster due to caching
3. Batch estimation uses single llm-optimizer calls per config
4. For large models, ensure sufficient RAM for analysis
5. GPU type matching: Use exact names (H100, A100, L40S, etc.) from gpu_library.py


# ==============================================================================
# BENTOML LLM-OPTIMIZER DOCUMENTATION
# ==============================================================================

For detailed information about llm-optimizer, see:
https://github.com/bentoml/llm-optimizer

Key capabilities:
- Latency prediction based on model architecture
- Multi-GPU and tensor parallelism support
- GPU-specific performance analysis
- Batch size impact modeling
- Sequence length sensitivity analysis
"""
