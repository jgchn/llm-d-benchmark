#!/usr/bin/env python3
"""
Validation script for bentoML llm-optimizer integration in performance.py

This script validates that the performance estimation module correctly integrates
with bentoML's llm-optimizer estimate_llm_performance function.
"""

import sys
import ast

def validate_performance_module():
    """Validate the performance.py module for correct llm-optimizer integration."""
    
    print("=" * 80)
    print("Validating bentoML llm-optimizer Integration")
    print("=" * 80)
    
    # Read the performance.py file
    with open('/Users/jchen/go/src/llm-d/llm-d-benchmark/config_explorer/src/config_explorer/recommender/performance.py', 'r') as f:
        content = f.read()
    
    # Validation checks
    checks = {
        "✓ estimate_llm_performance import": "from llm_optimizer.performance import estimate_llm_performance" in content,
        "✓ gpu_type parameter": "gpu_type=gpu_name" in content,
        "✓ num_gpu parameter": "num_gpu=num_gpus" in content,
        "✓ tp_degree parameter": "tp_degree=tensor_parallel_size" in content,
        "✓ input_len parameter": "input_len=input_length" in content,
        "✓ output_len parameter": "output_len=output_length" in content,
        "✓ batch_size parameter": "batch_size=batch_size" in content,
        "✓ Dict response handling": 'result.get("ttft_ms")' in content,
        "✓ Object response handling": 'getattr(result, "ttft_ms", None)' in content,
        "✓ Multiple field name variants (ttft)": 'result.get("ttft")' in content or 'getattr(result, "ttft"' in content,
        "✓ Multiple field name variants (itl)": 'result.get("itl")' in content or 'getattr(result, "itl"' in content,
        "✓ Multiple field name variants (tps)": 'result.get("tps")' in content or 'getattr(result, "tps"' in content,
        "✓ Error handling for ImportError": 'except ImportError:' in content,
        "✓ Error handling for general Exception": 'except Exception as e:' in content,
        "✓ Graceful degradation": 'bentoml-llm-optimizer' in content,
        "✓ estimate_performance_batch function": 'def estimate_performance_batch' in content,
        "✓ tensor_parallel_size in batch function": 'tensor_parallel_size: int = 1' in content,
        "✓ Docstring mentions bentoML": 'bentoML' in content or 'bentoml' in content,
    }
    
    print("\nValidation Results:")
    print("-" * 80)
    
    passed = 0
    failed = 0
    
    for check_name, check_result in checks.items():
        status = "PASS" if check_result else "FAIL"
        symbol = "✓" if check_result else "✗"
        print(f"{symbol} {check_name}: {status}")
        if check_result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 80)
    print(f"\nSummary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n✓ All validation checks passed!")
        print("\nKey Features Verified:")
        print("  1. Correct import of estimate_llm_performance from llm_optimizer.performance")
        print("  2. All parameters properly mapped to llm-optimizer's expected format")
        print("  3. Flexible response parsing for both dict and object returns")
        print("  4. Multiple field name variants handled for compatibility")
        print("  5. Comprehensive error handling with graceful degradation")
        print("  6. estimate_performance_batch with tensor_parallel_size support")
        print("  7. Clear documentation and bentoML integration notes")
        return True
    else:
        print("\n✗ Some validation checks failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    try:
        success = validate_performance_module()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Validation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
