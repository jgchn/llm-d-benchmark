"""
ModelArchitecture: Encapsulates model information and configuration.

Provides a high-level interface to model details fetched from HuggingFace Hub.
Reuses capacity_planner functions for memory and KV cache calculations.
"""

from dataclasses import dataclass
from typing import Optional
from huggingface_hub import ModelInfo
from transformers import AutoConfig

from ..capacity_planner import (
    get_model_info_from_hf,
    get_model_config_from_hf,
    get_text_config,
    model_total_params,
    model_memory_req,
    kv_cache_req,
    max_context_len,
    find_possible_tp,
    KVCacheDetail,
    inference_dtype,
    inference_dtype_byte,
    is_quantized,
    get_quant_method,
)


@dataclass
class ModelArchitecture:
    """
    Encapsulates LLM model information and configuration.

    Fetches model info from HuggingFace and provides convenient access to
    model properties such as parameters, memory requirements, and KV cache sizes.
    
    Reuses KVCacheDetail from capacity_planner for all KV cache calculations.
    """

    model_id: str
    """HuggingFace model ID (e.g., 'meta-llama/Llama-3.1-8B')."""

    model_info: ModelInfo
    """ModelInfo object from HuggingFace Hub."""

    model_config: AutoConfig
    """Model configuration from transformers library."""

    @classmethod
    def from_hf(cls, model_id: str, hf_token: Optional[str] = None) -> "ModelArchitecture":
        """
        Load model architecture from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID (e.g., 'meta-llama/Llama-3.1-8B')
            hf_token: Optional HuggingFace API token for gated models

        Returns:
            ModelArchitecture instance

        Raises:
            Various exceptions from transformers/huggingface_hub if model not found
        """
        model_info = get_model_info_from_hf(model_id, hf_token)
        model_config = get_model_config_from_hf(model_id, hf_token)
        return cls(model_id, model_info, model_config)

    def __init__(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize ModelArchitecture by fetching from HuggingFace.

        Args:
            model_id: HuggingFace model ID
            hf_token: Optional HuggingFace API token
        """
        self.model_id = model_id
        self.model_info = get_model_info_from_hf(model_id, hf_token)
        self.model_config = get_model_config_from_hf(model_id, hf_token)

    @property
    def num_parameters(self) -> int:
        """Total model parameters."""
        return model_total_params(self.model_info)

    @property
    def num_parameters_billions(self) -> float:
        """Total model parameters in billions."""
        return self.num_parameters / 1e9

    @property
    def model_name(self) -> str:
        """Extracted model name from model_id."""
        return self.model_id.split("/")[-1]

    @property
    def is_quantized(self) -> bool:
        """Check if model is quantized."""
        return is_quantized(self.model_config)
    
    @property
    def inferred_dtype(self) -> str:
        """
        Get the model's inferred data type for weights.
        
        Automatically detected from model config using capacity_planner's inference_dtype().
        Examples: 'torch.float32', 'torch.float16', 'torch.bfloat16', 'int4', 'int8', etc.
        """
        return inference_dtype(self.model_config)
    
    @property
    def inferred_precision_name(self) -> str:
        """
        Get human-readable precision name.
        
        Extracts precision from the inferred dtype.
        Examples: 'fp32', 'fp16', 'bf16', 'int4', 'int8', 'fp8'
        """
        dtype_str = self.inferred_dtype.lower()
        
        # Map torch dtype strings to common names
        if 'float32' in dtype_str or 'float32' in dtype_str:
            return 'fp32'
        elif 'float16' in dtype_str:
            return 'fp16'
        elif 'bfloat16' in dtype_str:
            return 'bf16'
        elif 'float8' in dtype_str or 'e5m2' in dtype_str or 'e4m3' in dtype_str:
            return 'fp8'
        elif 'int4' in dtype_str or 'int4' in dtype_str:
            return 'int4'
        elif 'int8' in dtype_str:
            return 'int8'
        else:
            # For quantized models, get from quant method
            if self.is_quantized:
                quant_method = get_quant_method(self.model_config)
                if quant_method:
                    return quant_method.lower()
            return dtype_str
    
    @property
    def inferred_kv_cache_dtype(self) -> str:
        """
        Get the model's inferred KV cache data type.
        
        For inference, KV cache might use a different dtype than model weights.
        This is automatically inferred from the model config.
        """
        return inference_dtype(self.model_config)

    def model_memory_gb(self, precision: str = None) -> float:
        """
        Calculate model weights memory requirement.
        
        If precision is not specified, uses capacity_planner's inference_dtype()
        to automatically infer the model's precision from config.

        Args:
            precision: Model precision (e.g., 'fp32', 'fp16', 'int4').
                      If None, automatically inferred from model config.

        Returns:
            Memory in GB
        """
        return model_memory_req(self.model_info, self.model_config)

    def _get_kv_cache_detail(self, context_length: int = 1, batch_size: int = 1) -> KVCacheDetail:
        """
        Get KVCacheDetail object for the model.
        
        This creates a KVCacheDetail object which handles all KV cache calculations
        including attention type detection, per-token memory, and batch calculations.
        """
        return KVCacheDetail(self.model_info, self.model_config, context_length, batch_size)

    def kv_cache_memory_gb(
        self,
        context_length: int = 1,
        batch_size: int = 1,
    ) -> float:
        """
        Calculate KV cache memory requirement using KVCacheDetail.

        Args:
            context_length: Input context length in tokens
            batch_size: Batch size (number of concurrent requests)

        Returns:
            Memory in GB
        """
        kv_detail = self._get_kv_cache_detail(context_length, batch_size)
        return kv_detail.kv_cache_size_gb

    def get_kv_cache_detail(self, context_length: int = 1, batch_size: int = 1) -> KVCacheDetail:
        """
        Get detailed KV cache information using KVCacheDetail.
        
        This provides access to the full KVCacheDetail object which includes:
        - Attention type and mechanism details
        - Per-token memory calculations
        - Per-request KV cache size
        - Complete breakdown of KV cache memory usage
        
        Args:
            context_length: Input context length in tokens
            batch_size: Batch size
            
        Returns:
            KVCacheDetail object with all KV cache information
        """
        return self._get_kv_cache_detail(context_length, batch_size)

    def total_memory_gb(
        self,
        context_length: int = 1,
        batch_size: int = 1,
        precision: str = "fp32",
    ) -> float:
        """
        Calculate total memory (model + KV cache).

        Args:
            context_length: Input context length in tokens
            batch_size: Batch size
            precision: Model precision

        Returns:
            Total memory in GB
        """
        model_mem = self.model_memory_gb(precision)
        kv_mem = self.kv_cache_memory_gb(context_length, batch_size)
        return model_mem + kv_mem

    def max_context_length(self) -> int:
        """Get maximum context length supported by model."""
        return max_context_len(self.model_config)

    def possible_tensor_parallel_sizes(self) -> list[int]:
        """Get possible tensor parallelism (TP) values for this model."""
        return find_possible_tp(self.model_config)
