# KVCacheDetail Integration in GPU Recommender

## Overview

The GPU recommender module extensively reuses `KVCacheDetail` from the capacity_planner module to ensure accurate and consistent KV cache memory calculations. This document explains the integration points and how KVCacheDetail is leveraged.

## What is KVCacheDetail?

`KVCacheDetail` is a comprehensive dataclass from `capacity_planner.py` that encapsulates all information needed to calculate KV cache memory requirements. It handles:

### Key Features:
1. **Attention Type Detection**: Automatically identifies which attention mechanism the model uses:
   - MLA (Multi-head Latent Attention) - for DeepSeek models
   - MHA (Multi-head Attention) - standard attention
   - GQA (Grouped-Query Attention) - efficient variant
   - MQA (Multi-Query Attention) - ultra-efficient variant

2. **Per-Token Memory Calculation**: Computes memory used per token, accounting for:
   - Number of hidden layers
   - Head dimension
   - Number of key-value heads
   - Precision/data type (FP32, FP16, FP8, etc.)

3. **Batch Calculations**: Handles different batch sizes and context lengths

### Data Fields:
```python
class KVCacheDetail:
    # Model configuration details
    model: str
    attention_type: AttentionType
    kv_data_type: str
    precision_in_bytes: int
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dimension: int
    model_architecture: str
    
    # Calculated values
    num_attention_group: int
    per_token_memory_bytes: int          # Memory per single token
    per_request_kv_cache_bytes: int      # Memory for one request with full context
    per_request_kv_cache_gb: float       # Same in GB
    kv_cache_size_gb: float              # Total KV cache for batch
    
    # Workload parameters
    context_len: int
    batch_size: int
```

## Integration Points in Recommender

### 1. ModelArchitecture Class

The `ModelArchitecture` class uses KVCacheDetail internally:

```python
class ModelArchitecture:
    def _get_kv_cache_detail(self, context_length: int = 1, batch_size: int = 1) -> KVCacheDetail:
        """Creates a KVCacheDetail object for the model."""
        return KVCacheDetail(self.model_info, self.model_config, context_length, batch_size)
    
    def get_kv_cache_detail(self, context_length: int = 1, batch_size: int = 1) -> KVCacheDetail:
        """Public API to access detailed KV cache information."""
        return self._get_kv_cache_detail(context_length, batch_size)
    
    def kv_cache_memory_gb(self, context_length: int = 1, batch_size: int = 1) -> float:
        """Uses KVCacheDetail internally to calculate KV cache memory."""
        kv_detail = self._get_kv_cache_detail(context_length, batch_size)
        return kv_detail.kv_cache_size_gb
```

### 2. find_gpu_recommendations Function

The core recommendation algorithm uses KVCacheDetail:

```python
def find_gpu_recommendations(model, context_length, batch_size, ...):
    # Get KVCacheDetail for accurate per-request memory calculation
    kv_detail = model.get_kv_cache_detail(context_length, batch_size=1)
    
    for gpu_name, gpu_spec in GPU_LIBRARY.items():
        for tp in possible_tp_values:
            # Use per-request KV cache from KVCacheDetail
            per_request_kv = kv_detail.per_request_kv_cache_gb
            
            # Calculate max concurrent requests
            max_concurrent = int(available_memory / per_request_kv)
```

### 3. Public API

Users can access KVCacheDetail through ModelArchitecture:

```python
from config_explorer.recommender import ModelArchitecture, KVCacheDetail

model = ModelArchitecture("meta-llama/Llama-3.1-8B")

# Get detailed KV cache information
kv_detail = model.get_kv_cache_detail(context_length=4096, batch_size=4)

print(f"Attention Type: {kv_detail.attention_type}")
print(f"Per-Token Memory: {kv_detail.per_token_memory_bytes} bytes")
print(f"Per-Request KV Cache: {kv_detail.per_request_kv_cache_gb:.2f} GB")
print(f"Total Batch KV Cache: {kv_detail.kv_cache_size_gb:.2f} GB")
```

## Benefits of Using KVCacheDetail

1. **Accuracy**: Leverages battle-tested capacity_planner logic
2. **Consistency**: Same calculations used throughout the codebase
3. **Comprehensive**: Handles all attention mechanisms (MLA, MHA, GQA, MQA)
4. **Flexibility**: Easy to adjust context length and batch size
5. **Transparency**: Users can inspect per-token memory calculations
6. **No Duplication**: Avoids reimplementing complex KV cache logic

## Example: Detailed Model Analysis

```python
from config_explorer.recommender import ModelArchitecture

model = ModelArchitecture("meta-llama/Llama-3.1-70B")

# Get detailed KV cache information
kv_detail_2k = model.get_kv_cache_detail(context_length=2048)
kv_detail_4k = model.get_kv_cache_detail(context_length=4096)

print(f"Model: {model.model_id}")
print(f"Parameters: {model.num_parameters_billions:.1f}B")
print(f"Model Memory: {model.model_memory_gb():.2f} GB\n")

print(f"Attention Type: {kv_detail_2k.attention_type}")
print(f"KV Data Type: {kv_detail_2k.kv_data_type}")
print(f"Per-Token Memory: {kv_detail_2k.per_token_memory_bytes} bytes\n")

print(f"Context Length: 2048")
print(f"  Per-Request KV Cache: {kv_detail_2k.per_request_kv_cache_gb:.3f} GB")
print(f"  Batch Size 8: {kv_detail_2k.kv_cache_size_gb * 8:.2f} GB\n")

print(f"Context Length: 4096")
print(f"  Per-Request KV Cache: {kv_detail_4k.per_request_kv_cache_gb:.3f} GB")
print(f"  Batch Size 8: {kv_detail_4k.kv_cache_size_gb * 8:.2f} GB")
```

## Implementation Details

### How KVCacheDetail Calculates Memory

For **non-MLA** models (standard attention):
```
per_token_memory = num_layers × 2 × head_dim × num_kv_heads × precision_bytes
```

For **MLA** models (DeepSeek):
```
per_token_memory = num_layers × (kv_lora_rank + qk_rope_head_dim) × precision_bytes
```

Then:
```
per_request_kv_cache = per_token_memory × context_length
kv_cache_for_batch = per_request_kv_cache × batch_size
```

### Attention Type Detection

KVCacheDetail automatically detects:
- **MLA**: DeepseekV3ForCausalLM, DeepseekV2ForCausalLM
- **MQA**: When `num_key_value_heads == 1`
- **MHA**: When `num_key_value_heads == num_attention_heads`
- **GQA**: When `1 < num_key_value_heads < num_attention_heads`

## Performance Characteristics

Creating a KVCacheDetail object is lightweight:
- One-time HuggingFace config load per model
- Quick calculation of per-token memory
- No heavy computations or external calls

The recommender reuses the single KVCacheDetail object created in `find_gpu_recommendations()` for all GPU configurations, minimizing overhead.

## Testing

The integration is tested through:
1. `TestModelArchitecture::test_kv_cache_calculation` - Verifies KV cache values scale correctly
2. `TestGPURecommendation::test_kv_cache_memory_calculation_in_recommendation` - Ensures recommendations use accurate values
3. Manual testing with various models (Qwen, Llama, DeepSeek, etc.)

## Future Enhancements

Potential improvements:
1. **Caching**: Cache KVCacheDetail objects for different context lengths
2. **Optimization**: Detect and suggest optimal context lengths
3. **Analysis**: Provide breakdown of KV cache by layer/head
4. **Visualization**: Create graphs of memory vs. context length
5. **Precision Selection**: Recommend KV cache quantization options

## Related Functions in Capacity Planner

The recommender also reuses these capacity_planner functions:
- `model_total_params()` - Get model parameter count
- `model_memory_req()` - Calculate model weight memory
- `find_possible_tp()` - Find valid tensor parallelism values
- `max_context_len()` - Get maximum supported context length

## Conclusion

KVCacheDetail is a critical component enabling the GPU recommender to provide accurate and reliable recommendations. By reusing this battle-tested module, we ensure consistency, accuracy, and maintainability across the config-explorer library.
