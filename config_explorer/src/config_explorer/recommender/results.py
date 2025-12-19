"""
Recommendation Results: Structured output and formatting for GPU recommendations.

Provides human-readable output for recommendation results with performance estimates.
"""

from dataclasses import dataclass, field
from typing import Optional
from tabulate import tabulate

from .recommendation import GPURecommendation
from .performance import PerformanceEstimate


@dataclass
class RecommendationWithPerformance:
    """A GPU recommendation with estimated performance metrics."""
    
    recommendation: GPURecommendation
    """GPU recommendation (GPU + TP config)."""
    
    performance: Optional[PerformanceEstimate] = None
    """Estimated performance metrics."""
    
    rank: int = 0
    """Ranking score for sorting (lower is better)."""
    
    def __repr__(self) -> str:
        gpu_name = self.recommendation.gpu_spec.name
        tp = self.recommendation.tensor_parallel_size
        mem = self.recommendation.memory_per_gpu_gb
        
        config_str = f"{gpu_name}×{tp}" if tp > 1 else gpu_name
        status = "✓" if self.recommendation.model_fits else "✗"
        
        perf_str = ""
        if self.performance:
            if self.performance.is_valid():
                perf_str = f" | {self.performance}"
            else:
                perf_str = f" | {self.performance.error}"
        
        return f"{status} {config_str:15} ({mem:6.1f}GB/GPU){perf_str}"


@dataclass
class RecommendationResult:
    """Complete recommendation result with all configurations and summary."""
    
    model_id: str
    """Model ID that was analyzed."""
    
    input_length: int
    """Input prompt length used."""
    
    output_length: int
    """Output generation length used."""
    
    precision: str
    """Model precision used."""
    
    recommendations: list[RecommendationWithPerformance] = field(default_factory=list)
    """Sorted list of recommendations with performance estimates."""
    
    total_recommendations: int = 0
    """Total number of recommendations."""
    
    fitting_recommendations: int = 0
    """Number of recommendations where model fits in memory."""
    
    def add_recommendation(
        self,
        rec: GPURecommendation,
        perf: Optional[PerformanceEstimate] = None,
    ) -> None:
        """Add a recommendation with optional performance estimate."""
        rec_with_perf = RecommendationWithPerformance(
            recommendation=rec,
            performance=perf,
        )
        self.recommendations.append(rec_with_perf)
        self.total_recommendations += 1
        if rec.model_fits:
            self.fitting_recommendations += 1
    
    def filter_fitting_only(self) -> "RecommendationResult":
        """Return a new result with only fitting recommendations."""
        new_result = RecommendationResult(
            model_id=self.model_id,
            input_length=self.input_length,
            output_length=self.output_length,
            precision=self.precision,
        )
        for rec in self.recommendations:
            if rec.recommendation.model_fits:
                new_result.add_recommendation(
                    rec.recommendation,
                    rec.performance,
                )
        return new_result
    
    def filter_by_gpu_name(self, gpu_name: str) -> "RecommendationResult":
        """Filter recommendations by GPU name."""
        new_result = RecommendationResult(
            model_id=self.model_id,
            input_length=self.input_length,
            output_length=self.output_length,
            precision=self.precision,
        )
        for rec in self.recommendations:
            if rec.recommendation.gpu_spec.name == gpu_name:
                new_result.add_recommendation(
                    rec.recommendation,
                    rec.performance,
                )
        return new_result
    
    def sort_by_cost(self) -> None:
        """Sort recommendations by total cost (GPU count × memory)."""
        self.recommendations.sort(
            key=lambda r: (
                r.recommendation.total_gpus,
                r.recommendation.gpu_spec.memory_gb,
            )
        )
    
    def sort_by_performance(self) -> None:
        """Sort by TTFT (time to first token), with valid estimates first."""
        def sort_key(rec):
            if rec.performance and rec.performance.is_valid():
                if rec.performance.time_to_first_token_ms is not None:
                    return (0, rec.performance.time_to_first_token_ms)
            return (1, float('inf'))
        
        self.recommendations.sort(key=sort_key)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"GPU Recommendation Report")
        lines.append("=" * 80)
        lines.append(f"Model: {self.model_id}")
        lines.append(f"Precision: {self.precision}")
        lines.append(f"Input Length: {self.input_length} tokens")
        lines.append(f"Output Length: {self.output_length} tokens")
        lines.append("")
        lines.append(f"Found {self.total_recommendations} viable configurations "
                     f"({self.fitting_recommendations} fit in memory)")
        lines.append("")
        
        if not self.recommendations:
            lines.append("No recommendations found!")
            return "\n".join(lines)
        
        # Create table data
        table_data = []
        for i, rec in enumerate(self.recommendations, 1):
            gpu = rec.recommendation.gpu_spec
            tp = rec.recommendation.tensor_parallel_size
            mem_per_gpu = rec.recommendation.memory_per_gpu_gb
            avail_kv = rec.recommendation.available_memory_for_kv_gb
            status = "YES" if rec.recommendation.model_fits else "NO"
            
            row = [
                str(i),
                f"{gpu.name}",
                str(tp),
                f"{mem_per_gpu:.2f}",
                f"{avail_kv:.2f}",
                f"{rec.recommendation.max_concurrent_requests}",
                status,
            ]
            
            # Add performance if available
            if rec.performance and rec.performance.is_valid():
                ttft = f"{rec.performance.time_to_first_token_ms:.1f}ms" \
                    if rec.performance.time_to_first_token_ms else "N/A"
                itl = f"{rec.performance.inter_token_latency_ms:.2f}ms" \
                    if rec.performance.inter_token_latency_ms else "N/A"
                tps = f"{rec.performance.throughput_tokens_per_sec:.1f}" \
                    if rec.performance.throughput_tokens_per_sec else "N/A"
                row.extend([ttft, itl, tps])
            
            table_data.append(row)
        
        # Create table headers
        headers = [
            "#",
            "GPU",
            "TP",
            "Model Memory\n(GB/GPU)",
            "Available for\nKV Cache (GB)",
            "Max Concurrent\nRequests",
            "Fits\nMemory",
        ]
        
        # Add performance headers if any estimate is valid
        has_perf = any(
            rec.performance and rec.performance.is_valid()
            for rec in self.recommendations
        )
        if has_perf:
            headers.extend(["TTFT", "ITL", "Throughput\n(tok/s)"])
        
        table_str = tabulate(
            table_data,
            headers=headers,
            tablefmt="grid",
            stralign="center",
        )
        
        lines.append("Recommendations:")
        lines.append(table_str)
        lines.append("")
        
        # Add recommendations
        lines.append("Ranked Recommendations (by feasibility and performance):")
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "precision": self.precision,
            "total_recommendations": self.total_recommendations,
            "fitting_recommendations": self.fitting_recommendations,
            "recommendations": [
                {
                    "gpu_name": rec.recommendation.gpu_spec.name,
                    "gpu_memory_gb": rec.recommendation.gpu_spec.memory_gb,
                    "tensor_parallel_size": rec.recommendation.tensor_parallel_size,
                    "total_gpus": rec.recommendation.total_gpus,
                    "model_fits": rec.recommendation.model_fits,
                    "memory_per_gpu_gb": rec.recommendation.memory_per_gpu_gb,
                    "available_memory_for_kv_gb": rec.recommendation.available_memory_for_kv_gb,
                    "max_concurrent_requests": rec.recommendation.max_concurrent_requests,
                    "max_batch_size": rec.recommendation.max_batch_size,
                    "performance": {
                        "ttft_ms": rec.performance.time_to_first_token_ms,
                        "itl_ms": rec.performance.inter_token_latency_ms,
                        "throughput_tok_s": rec.performance.throughput_tokens_per_sec,
                        "error": rec.performance.error,
                    } if rec.performance else None,
                }
                for rec in self.recommendations
            ],
        }
