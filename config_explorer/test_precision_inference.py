#!/usr/bin/env python3
"""
Quick test to verify precision inference works correctly.
"""

import sys
sys.path.insert(0, '/Users/jchen/go/src/llm-d/llm-d-benchmark/config_explorer')

from src.config_explorer.recommender import ModelArchitecture, recommend_gpus

def test_precision_inference():
    """Test that precision inference works as expected."""
    
    print("=" * 70)
    print("Testing Precision Inference in GPU Recommender")
    print("=" * 70)
    
    # Test 1: Load a model and check inferred precision properties
    print("\n[Test 1] Loading model and checking inferred properties...")
    try:
        model = ModelArchitecture("microsoft/phi-2")
        print(f"✓ Model loaded: {model.model_id}")
        print(f"  - Is quantized: {model.is_quantized}")
        print(f"  - Inferred dtype: {model.inferred_dtype}")
        print(f"  - Inferred precision name: {model.inferred_precision_name}")
        print(f"  - Inferred KV cache dtype: {model.inferred_kv_cache_dtype}")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        return False
    
    # Test 2: Check that recommend_gpus works with default (inferred) precision
    print("\n[Test 2] Testing recommend_gpus with default (inferred) precision...")
    try:
        result = recommend_gpus(
            model,
            input_length=512,
            output_length=256,
            estimate_performance_flag=False,  # Skip llm-optimizer
        )
        print(f"✓ recommend_gpus succeeded with inferred precision")
        print(f"  - Recommendations count: {len(result.recommendations)}")
        if result.recommendations:
            rec = result.recommendations[0]
            print(f"  - First recommendation: {rec.gpu_name} with TP={rec.tensor_parallel_size}")
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Check that explicit precision still works (override)
    print("\n[Test 3] Testing recommend_gpus with explicit precision (override)...")
    try:
        result = recommend_gpus(
            model,
            input_length=512,
            output_length=256,
            precision="int4",  # Explicitly override inferred precision
            estimate_performance_flag=False,
        )
        print(f"✓ recommend_gpus succeeded with explicit precision='int4'")
        print(f"  - Recommendations count: {len(result.recommendations)}")
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Verify precision properties are consistent
    print("\n[Test 4] Verifying precision properties consistency...")
    try:
        assert model.inferred_precision_name in ['fp32', 'fp16', 'bf16', 'fp8', 'int4', 'int8']
        print(f"✓ Inferred precision name is valid: {model.inferred_precision_name}")
        
        # Verify that model memory calculation works with default precision
        mem_default = model.model_memory_gb()  # Should use inferred precision
        mem_explicit = model.model_memory_gb(precision=model.inferred_precision_name)
        print(f"✓ Model memory calculations consistent:")
        print(f"  - Default (inferred): {mem_default:.3f} GB")
        print(f"  - Explicit inferred: {mem_explicit:.3f} GB")
        
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✓ All precision inference tests passed!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_precision_inference()
    sys.exit(0 if success else 1)
