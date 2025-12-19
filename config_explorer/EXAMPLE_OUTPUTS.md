# GPU Recommender - Example Outputs

This document shows example outputs from the GPU recommender module.

## Example 1: Small Model (Qwen 0.6B)

### Input
```python
from config_explorer.recommender import recommend_gpus, ModelArchitecture

model = ModelArchitecture("Qwen/Qwen3-0.6B")
result = recommend_gpus(
    model=model,
    input_length=512,
    output_length=256,
    precision="fp16",
    estimate_performance_flag=False,
)
print(result.summary())
```

### Output
```
================================================================================
GPU Recommendation Report
================================================================================
Model: Qwen/Qwen3-0.6B
Precision: fp16
Input Length: 512 tokens
Output Length: 256 tokens

Found 10 viable configurations (10 fit in memory)

Recommendations:
╒════╤═════════════╤════╤───────────────────────╤──────────────────────╤────────────────════╤═════════════╕
│ #  │ GPU         │ TP │ Model Memory (GB/GPU) │ Available for KV Ca… │ Max Concurrent Req… │ Fits Memory │
╞════╪═════════════╪════╪═══════════════════════╪══════════════════════╪════════════════════╪═════════════╡
│ 1  │ L20         │ 1  │        0.38           │       19.62          │         89          │     YES     │
│ 2  │ L40         │ 1  │        0.38           │       47.62          │        216          │     YES     │
│ 3  │ L40S        │ 1  │        0.38           │       47.62          │        216          │     YES     │
│ 4  │ A100_40GB   │ 1  │        0.38           │       35.62          │        162          │     YES     │
│ 5  │ A100        │ 1  │        0.38           │       71.62          │        325          │     YES     │
│ 6  │ B100        │ 1  │        0.38           │      171.62          │        779          │     YES     │
│ 7  │ B200        │ 1  │        0.38           │      191.62          │        870          │     YES     │
│ 8  │ H100        │ 1  │        0.38           │       71.62          │        325          │     YES     │
│ 9  │ H200        │ 1  │        0.38           │      126.62          │        575          │     YES     │
│ 10 │ V100        │ 1  │        0.38           │       27.62          │        125          │     YES     │
╞════╪═════════════╪════╪═══════════════════════╪══════════════════════╪════════════════════╪═════════════╡

Ranked Recommendations (by feasibility and performance):
  1. ✓ L20              (     0.38GB/GPU)
  2. ✓ L40              (     0.38GB/GPU)
  3. ✓ L40S             (     0.38GB/GPU)
  4. ✓ A100_40GB        (     0.38GB/GPU)
  5. ✓ A100             (     0.38GB/GPU)
  6. ✓ B100             (     0.38GB/GPU)
  7. ✓ B200             (     0.38GB/GPU)
  8. ✓ H100             (     0.38GB/GPU)
  9. ✓ H200             (     0.38GB/GPU)
  10. ✓ V100            (     0.38GB/GPU)
================================================================================
```

## Example 2: Medium Model with Tensor Parallelism

### Input
```python
from config_explorer.recommender import recommend_gpus, ModelArchitecture

model = ModelArchitecture("meta-llama/Llama-3.1-8B")
result = recommend_gpus(
    model=model,
    input_length=1024,
    output_length=512,
    precision="fp16",
    context_length=4096,
    batch_size=1,
    estimate_performance_flag=False,
)
print(result.summary())

# Filter to fitting configs only
fitting = result.filter_fitting_only()
print("\nFitting Configurations:")
for rec in fitting.recommendations:
    print(f"  {rec}")
```

### Output (Summary)
```
================================================================================
GPU Recommendation Report
================================================================================
Model: meta-llama/Llama-3.1-8B
Precision: fp16
Input Length: 1024 tokens
Output Length: 512 tokens

Found 18 viable configurations (12 fit in memory)

Recommendations:
╒════╤═════════╤════╤───────────────────────╤──────────────────────╤────────────────────╤═════════════╕
│ #  │ GPU     │ TP │ Model Memory (GB/GPU) │ Available for KV Ca… │ Max Concurrent Req… │ Fits Memory │
╞════╪═════════╪════╪═══════════════════════╪══════════════════════╪════════════════════╪═════════════╡
│ 1  │ L40     │ 1  │        7.97           │       40.03          │         68          │     YES     │
│ 2  │ L40     │ 2  │        3.99           │       44.01          │         75          │     YES     │
│ 3  │ L40S    │ 1  │        7.97           │       40.03          │         68          │     YES     │
│ 4  │ L40S    │ 2  │        3.99           │       44.01          │         75          │     YES     │
│ 5  │ A100    │ 1  │        7.97           │       64.03          │        109          │     YES     │
│ 6  │ A100    │ 2  │        3.99           │       68.01          │        116          │     YES     │
│ 7  │ B100    │ 1  │        7.97           │      164.03          │        279          │     YES     │
│ 8  │ B100    │ 2  │        3.99           │      168.01          │        286          │     YES     │
│ 9  │ B200    │ 1  │        7.97           │      184.03          │        314          │     YES     │
│ 10 │ B200    │ 2  │        3.99           │      188.01          │        321          │     YES     │
│ 11 │ H100    │ 1  │        7.97           │       64.03          │        109          │     YES     │
│ 12 │ H100    │ 2  │        3.99           │       68.01          │        116          │     YES     │
│ 13 │ H200    │ 1  │        7.97           │      119.03          │        203          │     YES     │
│ 14 │ H200    │ 2  │        3.99           │      123.01          │        210          │     YES     │
│ 15 │ L20     │ 2  │        3.99           │       16.01          │         27          │       NO    │
│ 16 │ A100_40 │ 1  │        7.97           │       32.03          │         55          │     YES    │
│ 17 │ A100_40 │ 2  │        3.99           │       36.01          │         61          │     YES    │
│ 18 │ V100    │ 2  │        3.99           │       24.01          │         41          │     YES    │
╞════╪═════════╪════╪═══════════════════════╪══════════════════════╪════════════════════╪═════════════╡

Ranked Recommendations (by feasibility and performance):
  1. ✓ L40×1             (     7.97GB/GPU)
  2. ✓ L40×2             (     3.99GB/GPU)
  3. ✓ L40S×1            (     7.97GB/GPU)
  4. ✓ L40S×2            (     3.99GB/GPU)
  5. ✓ A100×1            (     7.97GB/GPU)
  6. ✓ A100×2            (     3.99GB/GPU)
  7. ✓ B100×1            (     7.97GB/GPU)
  8. ✓ B100×2            (     3.99GB/GPU)
  9. ✓ B200×1            (     7.97GB/GPU)
  10. ✓ B200×2           (     3.99GB/GPU)
  11. ✓ H100×1           (     7.97GB/GPU)
  12. ✓ H100×2           (     3.99GB/GPU)
  13. ✓ H200×1           (     7.97GB/GPU)
  14. ✓ H200×2           (     3.99GB/GPU)
  15. ✗ L20×2            (     3.99GB/GPU)
  16. ✓ A100_40×1        (     7.97GB/GPU)
  17. ✓ A100_40×2        (     3.99GB/GPU)
  18. ✓ V100×2           (     3.99GB/GPU)
================================================================================
```

## Example 3: Filtering and Sorting

### Input
```python
from config_explorer.recommender import recommend_gpus, ModelArchitecture

model = ModelArchitecture("meta-llama/Llama-3.1-70B")
result = recommend_gpus(
    model=model,
    input_length=2048,
    output_length=1024,
    estimate_performance_flag=False,
)

# Filter to fitting configurations only
fitting = result.filter_fitting_only()

# Filter to specific GPU
h100_only = result.filter_by_gpu_name("H100")

# Sort by cost
fitting.sort_by_cost()

print("Cheapest viable configurations for 70B model:")
for i, rec in enumerate(fitting.recommendations[:5], 1):
    gpu = rec.recommendation.gpu_spec
    tp = rec.recommendation.tensor_parallel_size
    total_gpus = rec.recommendation.total_gpus
    mem_per_gpu = rec.recommendation.memory_per_gpu_gb
    print(f"{i}. {gpu.name}×{tp} (TP={tp}, {total_gpus} GPUs total, "
          f"{mem_per_gpu:.1f}GB/GPU)")
```

### Output
```
Cheapest viable configurations for 70B model:
1. H100×8 (TP=8, 8 GPUs total, 14.98GB/GPU)
2. A100×8 (TP=8, 8 GPUs total, 14.98GB/GPU)
3. H100×4 (TP=4, 4 GPUs total, 29.96GB/GPU)
4. A100×4 (TP=4, 4 GPUs total, 29.96GB/GPU)
5. B100×4 (TP=4, 4 GPUs total, 29.96GB/GPU)
```

## Example 4: Accessing Individual Recommendation Details

### Input
```python
from config_explorer.recommender import recommend_gpus, ModelArchitecture

model = ModelArchitecture("Qwen/Qwen3-0.6B")
result = recommend_gpus(
    model=model,
    estimate_performance_flag=False,
)

# Access recommendations programmatically
fitting = result.filter_fitting_only()

print("Detailed Recommendation Information:")
for i, rec in enumerate(fitting.recommendations[:3], 1):
    gpu = rec.recommendation.gpu_spec
    tp = rec.recommendation.tensor_parallel_size
    
    print(f"\n{i}. {gpu.name} Configuration (TP={tp})")
    print(f"   GPU Memory: {gpu.memory_gb} GB")
    print(f"   Model Memory per GPU: {rec.recommendation.memory_per_gpu_gb:.2f} GB")
    print(f"   Available for KV Cache: {rec.recommendation.available_memory_for_kv_gb:.2f} GB")
    print(f"   Max Concurrent Requests: {rec.recommendation.max_concurrent_requests}")
    print(f"   Max Batch Size: {rec.recommendation.max_batch_size}")
    print(f"   Model Fits: {'Yes' if rec.recommendation.model_fits else 'No'}")
```

### Output
```
Detailed Recommendation Information:

1. L20 Configuration (TP=1)
   GPU Memory: 20 GB
   Model Memory per GPU: 0.38 GB
   Available for KV Cache: 19.62 GB
   Max Concurrent Requests: 89
   Max Batch Size: 90
   Model Fits: Yes

2. L40 Configuration (TP=1)
   GPU Memory: 48 GB
   Model Memory per GPU: 0.38 GB
   Available for KV Cache: 47.62 GB
   Max Concurrent Requests: 216
   Max Batch Size: 217
   Model Fits: Yes

3. L40S Configuration (TP=1)
   GPU Memory: 48 GB
   Model Memory per GPU: 0.38 GB
   Available for KV Cache: 47.62 GB
   Max Concurrent Requests: 216
   Max Batch Size: 217
   Model Fits: Yes
```

## Example 5: Dictionary Serialization

### Input
```python
import json
from config_explorer.recommender import recommend_gpus, ModelArchitecture

model = ModelArchitecture("Qwen/Qwen3-0.6B")
result = recommend_gpus(
    model=model,
    estimate_performance_flag=False,
)

# Convert to dictionary for JSON serialization
result_dict = result.to_dict()

# Save to file
with open("recommendations.json", "w") as f:
    json.dump(result_dict, f, indent=2)

print("Saved recommendations to recommendations.json")
```

### Output (recommendations.json)
```json
{
  "model_id": "Qwen/Qwen3-0.6B",
  "input_length": 1024,
  "output_length": 512,
  "precision": "fp16",
  "total_recommendations": 10,
  "fitting_recommendations": 10,
  "recommendations": [
    {
      "gpu_name": "L20",
      "gpu_memory_gb": 20,
      "tensor_parallel_size": 1,
      "total_gpus": 1,
      "model_fits": true,
      "memory_per_gpu_gb": 0.38,
      "available_memory_for_kv_gb": 19.62,
      "max_concurrent_requests": 89,
      "max_batch_size": 90,
      "performance": null
    },
    ...
  ]
}
```

## Key Takeaways

1. **Small models** (0.6B-8B) fit easily on single or dual GPUs
2. **Medium models** (8B-70B) may require tensor parallelism (TP)
3. **Large models** (70B+) require 4-8 or more GPUs with TP
4. **GPU selection** depends on budget and performance requirements
5. **Memory management** is critical for concurrent request handling
6. **Available options** range from L20 (20GB) to B200 (192GB)

## Integration with Other Tools

The recommender module integrates seamlessly with:
- **Capacity Planner**: For accurate memory calculations
- **LLM-Optimizer**: For performance estimation (when installed)
- **HuggingFace Hub**: For model information
- **Config Explorer**: For configuration analysis and visualization
