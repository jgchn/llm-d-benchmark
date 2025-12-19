"""
README: GPU Recommender Module

The GPU Recommender module provides intelligent GPU and tensor parallelism (TP)
recommendations for serving LLM models with specified inference workloads.

## Features

- **GPU Library**: Comprehensive specifications for popular GPUs (A100, H100, L40, etc.)
- **Model Analysis**: Automatic model size and KV cache calculation using HuggingFace
- **Tensor Parallelism**: Finds viable TP configurations when model doesn't fit on single GPU
- **Performance Estimation**: Integrates with llm-optimizer for realistic inference estimates
- **Structured Output**: Human-readable recommendations with memory and performance details

## Quick Start

### Basic Usage

```python
from config_explorer.recommender import recommend_gpus, ModelArchitecture

# Load a model from HuggingFace
model = ModelArchitecture("Qwen/Qwen3-32B")

# Get GPU recommendations (precision is automatically inferred!)
result = recommend_gpus(
    model=model,
    input_length=1024,      # Expected input prompt length
    output_length=512,      # Expected output generation length
)

# Display results
print(result.summary())
```

### Advanced Usage

```python
from config_explorer.recommender import recommend_gpus, ModelArchitecture

model = ModelArchitecture("meta-llama/Llama-3.1-70B")

# Get recommendations with specific constraints
# Precision is automatically inferred from model config
result = recommend_gpus(
    model=model,
    input_length=2048,
    output_length=1024,
    context_length=4096,           # Max context for KV cache calculation
    batch_size=4,                   # Batch size for KV cache
    gpu_memory_utilization=0.9,     # 90% GPU memory utilization
    max_tensor_parallel=8,          # Don't use TP > 8
    max_gpus=16,                    # Don't recommend > 16 GPUs
    estimate_performance_flag=True, # Use llm-optimizer to estimate performance
)

# Override auto-inferred precision if needed
result = recommend_gpus(
    model=model,
    input_length=2048,
    output_length=1024,
    precision="fp8",                # Explicitly specify precision (overrides inference)
)

# Filter and sort results
fitting = result.filter_fitting_only()  # Only configs where model fits
fitting.sort_by_performance()           # Sort by TTFT

# Print results
print(fitting.summary())

# Access individual recommendations
for rec in fitting.recommendations:
    gpu_name = rec.recommendation.gpu_spec.name
    tp = rec.recommendation.tensor_parallel_size
    ttft = rec.performance.time_to_first_token_ms
    print(f"{gpu_name}×{tp}: TTFT={ttft:.1f}ms")
```

## API Reference

### Main Function

#### `recommend_gpus(model, input_length=1024, output_length=512, ...)`

Recommends GPUs for serving an LLM model.

**Parameters:**
- `model` (ModelArchitecture): Model to analyze
- `input_length` (int): Expected input prompt length (default: 1024)
- `output_length` (int): Expected output generation length (default: 512)
- `precision` (str, optional): Model precision - 'fp32', 'fp16', 'fp8', 'int4'. If None, automatically inferred from model config using capacity_planner (default: None)
- `context_length` (int): Context length for KV cache calc (default: 2048)
- `batch_size` (int): Batch size for KV cache (default: 1)
- `gpu_memory_utilization` (float): GPU memory utilization 0.0-1.0 (default: 0.9)
- `max_tensor_parallel` (int): Maximum TP to consider (default: model maximum)
- `estimate_performance_flag` (bool): Use llm-optimizer (default: True)
- `max_gpus` (int): Maximum total GPUs (default: no limit)

**Returns:** `RecommendationResult`

**Note on Precision Inference:**
When `precision=None` (the default), the function automatically infers the model's precision using capacity_planner's built-in functions, which analyze:
- Quantization configuration (e.g., int4, int8)
- Model's native data type (fp32, fp16, bfloat16)
- Special handling for MLA/MHA/GQA/MQA attention types

This eliminates the need to manually specify precision for most models. You can still explicitly provide a precision to override the inference.

### Classes

#### `ModelArchitecture`

Encapsulates model information from HuggingFace Hub.

**Constructor:**
```python
model = ModelArchitecture("meta-llama/Llama-3.1-8B", hf_token=None)
```

**Properties:**
- `num_parameters` (int): Total model parameters
- `num_parameters_billions` (float): Parameters in billions
- `model_name` (str): Model name extracted from ID
- `model_id` (str): Full HuggingFace model ID
- `is_quantized` (bool): Whether model is quantized (int4, int8, etc.)
- `inferred_dtype` (str): Inferred data type (e.g., 'torch.float16', 'int4')
- `inferred_precision_name` (str): Human-readable precision ('fp32', 'fp16', 'bf16', 'int4', 'int8', 'fp8')
- `inferred_kv_cache_dtype` (str): Inferred KV cache data type

**Methods:**
- `model_memory_gb(precision=None)` -> float: Model weights memory. If precision=None, uses inferred precision.
- `kv_cache_memory_gb(context_length=1, batch_size=1)` -> float: KV cache memory
- `total_memory_gb(context_length=1, batch_size=1, precision=None)` -> float: Total memory including KV cache
- `max_context_length()` -> int: Maximum context length
- `possible_tensor_parallel_sizes()` -> list[int]: Viable TP values
- `get_kv_cache_detail(context_length=1, batch_size=1)` -> KVCacheDetail: Detailed KV cache breakdown from capacity_planner

#### `GPURecommendation`

A single GPU + TP configuration recommendation.

**Attributes:**
- `gpu_spec` (GPUSpec): GPU specifications
- `tensor_parallel_size` (int): TP degree
- `total_gpus` (int): Total GPUs needed
- `model_fits` (bool): Whether model fits in GPU memory
- `memory_per_gpu_gb` (float): Model memory per GPU
- `available_memory_for_kv_gb` (float): Available KV cache memory
- `max_concurrent_requests` (int): Max concurrent requests

#### `GPUSpec`

GPU specifications and capabilities.

**Attributes:**
- `name` (str): GPU model name (e.g., 'H100')
- `memory_gb` (float): GPU memory in GB
- `tflops` (float): Peak FP32 TFLOPS
- `compute_capability` (str): Compute capability version
- `release_year` (int): Release year
- `manufacturer` (str): Manufacturer (default: 'NVIDIA')

#### `RecommendationResult`

Complete recommendation results with formatting.

**Methods:**
- `filter_fitting_only()` -> RecommendationResult: Get only fitting configs
- `filter_by_gpu_name(name)` -> RecommendationResult: Filter by GPU
- `sort_by_cost()`: Sort by total cost (GPUs × memory)
- `sort_by_performance()`: Sort by TTFT
- `summary()` -> str: Human-readable summary table
- `to_dict()` -> dict: Serialize to dictionary

#### `PerformanceEstimate`

Estimated inference performance metrics.

**Attributes:**
- `time_to_first_token_ms` (float): TTFT in milliseconds
- `inter_token_latency_ms` (float): ITL in milliseconds
- `throughput_tokens_per_sec` (float): Output throughput
- `max_batch_size` (int): Maximum batch size
- `max_concurrent_requests` (int): Maximum concurrent requests
- `error` (str): Error message if estimation failed

### GPU Library

#### `list_available_gpus()` -> list[str]

Get all available GPU names.

```python
from config_explorer.recommender import list_available_gpus
gpus = list_available_gpus()
# Output: ['A100', 'A100_40GB', 'B100', 'B200', 'H100', 'H200', 'L20', 'L40', 'L40S', 'V100']
```

#### `get_gpu_spec(gpu_name)` -> GPUSpec

Get specification for a GPU by name.

```python
from config_explorer.recommender import get_gpu_spec
h100 = get_gpu_spec("H100")
print(f"H100: {h100.memory_gb}GB, {h100.tflops} TFLOPS")
```

## Supported GPUs

- **H-series**: H100, H200 (Latest Hopper)
- **A-series**: A100, A100-40GB
- **B-series**: B100, B200 (Blackwell)
- **L-series**: L20, L40, L40S (Inference optimized)
- **V-series**: V100

## Integration with Capacity Planner

The recommender module extensively reuses functions from `capacity_planner`:
- **Model size calculation**: model_total_params, model_memory_req
- **KV cache memory estimation**: KVCacheDetail class with support for MLA/MHA/GQA/MQA attention
- **Tensor parallelism factors**: Proper TP degree calculations
- **Precision inference**: Automatic detection of model precision including quantization
- **Attention mechanism analysis**: Proper handling of different attention types

### Precision Inference

The module automatically infers model precision using capacity_planner's functions:

```python
from config_explorer.recommender import ModelArchitecture

model = ModelArchitecture("meta-llama/Llama-3.1-70B")

# Access inferred precision properties
print(f"Is quantized: {model.is_quantized}")          # False (for most non-GGUF models)
print(f"Inferred dtype: {model.inferred_dtype}")      # 'torch.float16'
print(f"Precision name: {model.inferred_precision_name}")  # 'fp16'
print(f"KV cache dtype: {model.inferred_kv_cache_dtype}")  # 'torch.float16'

# Use in recommendations with automatic precision
result = recommend_gpus(model)  # precision inferred automatically as 'fp16'

# Or override if needed
result = recommend_gpus(model, precision="int4")  # Force int4
```

**Supported Precision Types Detected:**
- `fp32` - Full precision floating point
- `fp16` - Half precision floating point
- `bf16` - Bfloat16 (brain float)
- `fp8` - 8-bit floating point
- `int4` - 4-bit integer quantization
- `int8` - 8-bit integer quantization

## Integration with llm-optimizer

When `estimate_performance_flag=True`, the module uses bentoML's llm-optimizer to estimate:
- **Time to First Token (TTFT)**: Latency before first output token
- **Inter-Token Latency (ITL)**: Time between output tokens
- **Output Throughput**: Tokens per second

Installation:
```bash
pip install bentoml-llm-optimizer
# or
pip install git+https://github.com/bentoml/llm-optimizer.git
```

The performance estimation automatically:
- Analyzes model architecture for latency prediction
- Considers GPU specifications and compute capabilities
- Accounts for tensor parallelism overhead
- Handles multi-GPU configurations
- Predicts maximum concurrent requests based on memory

See [PERFORMANCE_ESTIMATION.md](PERFORMANCE_ESTIMATION.md) for detailed examples and configuration options.

## Output Format

The `summary()` method produces a structured table with:

| # | GPU | TP | Model Memory | Available KV | Max Requests | Fits | TTFT | ITL | Throughput |
|---|-----|----|-----------|-----------|-----------|----|------|-----|-----------|
| 1 | H100 | 1 | 15.23 | 64.77 | 42 | YES | 45.2ms | 8.3ms | 125.4 tok/s |
| 2 | H100 | 2 | 7.62 | 72.38 | 87 | YES | 52.1ms | 7.9ms | 118.2 tok/s |

## Examples

### Example 1: Small Model with Auto Precision

```python
from config_explorer.recommender import recommend_gpus, ModelArchitecture

model = ModelArchitecture("Qwen/Qwen3-0.6B")
# Precision is automatically inferred as 'fp32'
result = recommend_gpus(model, estimate_performance_flag=False)
print(result.summary())
```

Output: Multiple GPU options fit (A100, L40, H100, etc.)

### Example 2: Large Model with TP and Auto Precision

```python
from config_explorer.recommender import recommend_gpus, ModelArchitecture

model = ModelArchitecture("meta-llama/Llama-3.1-70B")
# Precision is automatically inferred as 'fp16'
result = recommend_gpus(
    model,
    input_length=2048,
    output_length=1024,
    estimate_performance_flag=False,
)

# Filter to H100s only
h100_recs = result.filter_by_gpu_name("H100")
print(f"H100 configurations: {h100_recs.summary()}")
```

### Example 3: Quantized Model with Inferred Precision

```python
from config_explorer.recommender import recommend_gpus, ModelArchitecture

# Model with int4 quantization
model = ModelArchitecture("meta-llama/Llama-3.1-70B-instruct")
print(f"Is quantized: {model.is_quantized}")  # True if quantized
print(f"Inferred precision: {model.inferred_precision_name}")  # 'int4'

# Recommendations automatically use the inferred precision
result = recommend_gpus(
    model,
    input_length=1024,
    output_length=512,
)
print(result.summary())
```

### Example 4: Cost-Optimized Selection

```python
from config_explorer.recommender import recommend_gpus, ModelArchitecture

model = ModelArchitecture("meta-llama/Llama-3.1-8B")
result = recommend_gpus(model)

# Get only configurations that fit
fitting = result.filter_fitting_only()

# Sort by cost
fitting.sort_by_cost()

# Pick cheapest option
cheapest = fitting.recommendations[0]
print(f"Cheapest option: {cheapest}")
```

## Troubleshooting

### ModuleNotFoundError: No module named 'llm_optimizer'

Install llm-optimizer:
```bash
pip install git+https://github.com/bentoml/llm-optimizer.git
```

### Model not found

For gated models (e.g., Llama), you need HuggingFace token:
```bash
export HF_TOKEN=hf_your_token_here
```

Then pass it to ModelArchitecture:
```python
model = ModelArchitecture("meta-llama/Llama-3.1-8B", hf_token="hf_...")
```

### OutOfMemory when analyzing large models

This typically happens during model config loading. Make sure you have sufficient RAM
and try on a machine with more memory, or use a quantized model variant.

## Contributing

To add a new GPU to the library, edit `gpu_library.py`:

```python
GPU_LIBRARY["NewGPU"] = GPUSpec(
    name="NewGPU",
    memory_gb=80,
    tflops=1456,
    compute_capability="9.0",
    release_year=2024,
)
```

## License

Same as config_explorer (inherited from llm-d-benchmark)
"""
