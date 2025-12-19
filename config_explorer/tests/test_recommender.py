"""
Test suite for GPU recommender module.

Tests GPU recommendation logic, capacity planner integration, and output formatting.
"""

import pytest
from src.config_explorer.recommender import (
    ModelArchitecture,
    GPURecommendation,
    find_gpu_recommendations,
    filter_recommendations_by_budget,
    list_available_gpus,
    get_gpu_spec,
    recommend_gpus,
)


class TestGPULibrary:
    """Test GPU library functionality."""
    
    def test_list_available_gpus(self):
        """Test that GPU library has expected GPUs."""
        gpus = list_available_gpus()
        assert len(gpus) > 0
        assert "H100" in gpus
        assert "A100" in gpus
        assert "L40" in gpus
    
    def test_get_gpu_spec(self):
        """Test getting GPU specs."""
        h100 = get_gpu_spec("H100")
        assert h100.name == "H100"
        assert h100.memory_gb == 80
        assert h100.tflops > 0
        
        a100 = get_gpu_spec("A100")
        assert a100.memory_gb == 80
    
    def test_invalid_gpu_raises(self):
        """Test that invalid GPU name raises error."""
        with pytest.raises(ValueError):
            get_gpu_spec("InvalidGPU123")


class TestModelArchitecture:
    """Test ModelArchitecture class."""
    
    def test_load_small_model(self):
        """Test loading a small model from HuggingFace."""
        model = ModelArchitecture("Qwen/Qwen3-0.6B")
        
        assert model.model_id == "Qwen/Qwen3-0.6B"
        assert model.num_parameters > 0
        assert model.num_parameters_billions > 0.5
        assert model.model_name == "Qwen3-0.6B"
    
    def test_model_memory_calculation(self):
        """Test that model memory is calculated."""
        model = ModelArchitecture("Qwen/Qwen3-0.6B")
        
        # FP32 should be larger than FP16
        mem_fp32 = model.model_memory_gb("fp32")
        mem_fp16 = model.model_memory_gb("fp16")
        
        assert mem_fp32 > 0
        assert mem_fp16 > 0
        assert mem_fp32 > mem_fp16 * 1.5  # FP32 should be ~2x FP16
    
    def test_kv_cache_calculation(self):
        """Test KV cache memory calculation."""
        model = ModelArchitecture("Qwen/Qwen3-0.6B")
        
        # Larger context should use more memory
        kv_1k = model.kv_cache_memory_gb(context_length=1024, batch_size=1)
        kv_2k = model.kv_cache_memory_gb(context_length=2048, batch_size=1)
        
        assert kv_1k > 0
        assert kv_2k > kv_1k
        assert kv_2k / kv_1k == pytest.approx(2.0, rel=0.1)
    
    def test_total_memory_calculation(self):
        """Test total memory (model + KV cache)."""
        model = ModelArchitecture("Qwen/Qwen3-0.6B")
        
        total = model.total_memory_gb(context_length=2048, batch_size=1)
        model_mem = model.model_memory_gb()
        kv_mem = model.kv_cache_memory_gb(context_length=2048, batch_size=1)
        
        assert total == pytest.approx(model_mem + kv_mem)
    
    def test_max_context_length(self):
        """Test getting maximum context length."""
        model = ModelArchitecture("Qwen/Qwen3-0.6B")
        max_ctx = model.max_context_length()
        assert max_ctx > 0
        assert max_ctx > 2048  # Should support at least 2K
    
    def test_possible_tensor_parallel_sizes(self):
        """Test getting possible TP values."""
        model = ModelArchitecture("Qwen/Qwen3-0.6B")
        tp_values = model.possible_tensor_parallel_sizes()
        
        assert len(tp_values) > 0
        assert 1 in tp_values  # TP=1 should always be possible
        assert all(isinstance(tp, int) for tp in tp_values)


class TestGPURecommendation:
    """Test GPU recommendation logic."""
    
    def test_find_recommendations_for_small_model(self):
        """Test finding recommendations for a small model."""
        model = ModelArchitecture("Qwen/Qwen3-0.6B")
        
        recs = find_gpu_recommendations(
            model=model,
            context_length=2048,
            batch_size=1,
        )
        
        assert len(recs) > 0
        # Small model should fit on most GPUs
        fitting_recs = [r for r in recs if r.model_fits]
        assert len(fitting_recs) > 0
    
    def test_recommendations_sorted_by_memory(self):
        """Test that recommendations are sorted by GPU memory."""
        model = ModelArchitecture("Qwen/Qwen3-0.6B")
        
        recs = find_gpu_recommendations(
            model=model,
            context_length=2048,
        )
        
        # Check if sorted by GPU memory
        for i in range(len(recs) - 1):
            assert recs[i].gpu_spec.memory_gb <= recs[i+1].gpu_spec.memory_gb
    
    def test_kv_cache_memory_calculation_in_recommendation(self):
        """Test that KV cache memory is calculated correctly in recommendations."""
        model = ModelArchitecture("Qwen/Qwen3-0.6B")
        
        recs = find_gpu_recommendations(
            model=model,
            context_length=2048,
        )
        
        for rec in recs:
            # Available memory should be less than GPU memory
            assert rec.available_memory_for_kv_gb >= 0
            assert rec.available_memory_for_kv_gb <= rec.gpu_spec.memory_gb
            assert rec.memory_per_gpu_gb > 0


class TestRecommendationResult:
    """Test RecommendationResult formatting and filtering."""
    
    def test_recommend_gpus_api(self):
        """Test the main recommend_gpus API."""
        model = ModelArchitecture("Qwen/Qwen3-0.6B")
        
        result = recommend_gpus(
            model=model,
            input_length=512,
            output_length=256,
            precision="fp16",
            estimate_performance_flag=False,  # Skip performance estimation
        )
        
        assert result.model_id == model.model_id
        assert result.input_length == 512
        assert result.output_length == 256
        assert result.precision == "fp16"
        assert result.total_recommendations > 0
    
    def test_filter_fitting_only(self):
        """Test filtering recommendations to fitting only."""
        model = ModelArchitecture("Qwen/Qwen3-0.6B")
        
        result = recommend_gpus(
            model=model,
            estimate_performance_flag=False,
        )
        
        fitting = result.filter_fitting_only()
        
        # All filtered recommendations should fit
        for rec in fitting.recommendations:
            assert rec.recommendation.model_fits
    
    def test_summary_generation(self):
        """Test that summary can be generated without error."""
        model = ModelArchitecture("Qwen/Qwen3-0.6B")
        
        result = recommend_gpus(
            model=model,
            estimate_performance_flag=False,
        )
        
        summary = result.summary()
        
        assert isinstance(summary, str)
        assert model.model_id in summary
        assert "Recommendation" in summary
        assert len(summary) > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
