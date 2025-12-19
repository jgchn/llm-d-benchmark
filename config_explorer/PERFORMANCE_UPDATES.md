# Performance Estimation Updates

## Summary

Updated `performance.py` to properly integrate bentoML's `llm-optimizer` for accurate inference performance estimation.

## Changes Made

### 1. Function Signature Updates

**estimate_performance()**
- Now explicitly uses `estimate_llm_performance()` from bentoML's llm-optimizer
- Parameters properly mapped to llm-optimizer's expected format:
  - `gpu_name` → `gpu_type`
  - `num_gpus` → `num_gpu`
  - `tensor_parallel_size` → `tp_degree`
  - `input_length` → `input_len`
  - `output_length` → `output_len`

**estimate_performance_batch()**
- Added `tensor_parallel_size` parameter for consistency
- Enhanced docstring with examples
- Improved error handling

### 2. Response Parsing

Added flexible response parsing to handle both dict and object returns from llm-optimizer:

```python
# Handles multiple field name variations:
# ttft_ms or ttft
# itl_ms or itl  
# throughput or tps
# max_concurrent_requests or max_concurrent
```

This ensures compatibility with different llm-optimizer versions.

### 3. Error Handling

- Clear error messages if llm-optimizer not installed
- Graceful degradation (returns PerformanceEstimate with error, doesn't crash)
- Detailed logging for troubleshooting
- Per-configuration error tracking

### 4. Documentation

Created comprehensive documentation:
- `PERFORMANCE_ESTIMATION.md`: Detailed guide with examples
- Updated `README.md`: Links to performance documentation
- Inline docstrings with parameter descriptions and installation instructions

## Key Features

✅ **Direct Integration**: Uses bentoML's `estimate_llm_performance()` directly
✅ **Flexible Response Handling**: Works with multiple output formats
✅ **Graceful Degradation**: Works without llm-optimizer (performance estimation skipped)
✅ **Batch Estimation**: Efficient multi-config estimation with `estimate_performance_batch()`
✅ **Clear Documentation**: Comprehensive examples and troubleshooting guide

## Usage Example

```python
from config_explorer.recommender.performance import estimate_performance

estimate = estimate_performance(
    model_id="meta-llama/Llama-3.1-70B",
    gpu_name="H100",
    num_gpus=2,
    tensor_parallel_size=2,
    input_length=2048,
    output_length=512,
)

if estimate.is_valid():
    print(f"TTFT: {estimate.time_to_first_token_ms:.1f}ms")
    print(f"Throughput: {estimate.throughput_tokens_per_sec:.1f} tok/s")
```

## Files Modified

- `performance.py`: Updated function signatures, parameter mapping, and response parsing
- `README.md`: Updated integration section with installation instructions
- `PERFORMANCE_ESTIMATION.md`: New comprehensive documentation file

## Backward Compatibility

✅ All existing code continues to work
✅ Return types unchanged (PerformanceEstimate)
✅ Optional dependency (graceful fallback if not installed)
✅ API is more flexible with improved response parsing
