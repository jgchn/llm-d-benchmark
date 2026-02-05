# Safetensors Fallback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow the capacity planner to retrieve model parameters for models without safetensors metadata by using `accelerate.init_empty_weights()` to instantiate the model architecture without downloading weights.

**Architecture:** Add a fallback function `get_model_params_from_download()` that creates an empty model using `accelerate`, inspects `named_parameters()` to count params per dtype, and returns a dict compatible with the existing `safetensors.parameters` structure. The UI and CLI prompt users to opt-in when safetensors is unavailable.

**Tech Stack:** Python, accelerate, transformers, streamlit

---

## Task 1: Add accelerate dependency

**Files:**
- Modify: `config_explorer/pyproject.toml:11-21`

**Step 1: Add accelerate to dependencies**

In `pyproject.toml`, add `accelerate` to the dependencies list:

```toml
dependencies = [
    "accelerate>=0.25.0",
    "huggingface_hub>=0.34.4",
    "matplotlib>=3.10.5",
    "numpy>=2.3.2",
    "pandas>=2.3.1",
    "pydantic>=2.11.7",
    "PyYAML>=6.0.2",
    "scipy>=1.16.1",
    "transformers>=4.55.4",
    "llm-optimizer @ git+https://github.com/bentoml/llm-optimizer.git",
]
```

**Step 2: Commit**

```bash
git add config_explorer/pyproject.toml
git commit -m "feat: add accelerate dependency for safetensors fallback"
```

---

## Task 2: Implement `map_torch_dtype_to_safetensors()` helper

**Files:**
- Modify: `config_explorer/src/config_explorer/capacity_planner.py`
- Test: `config_explorer/tests/capacity_planner_test.py`

**Step 1: Write the failing test**

Add to `capacity_planner_test.py`:

```python
def test_map_torch_dtype_to_safetensors():
    """Tests mapping torch dtypes to safetensors dtype strings"""
    import torch

    # Test standard dtypes
    assert map_torch_dtype_to_safetensors(torch.float32, None) == "F32"
    assert map_torch_dtype_to_safetensors(torch.float16, None) == "F16"
    assert map_torch_dtype_to_safetensors(torch.bfloat16, None) == "BF16"
    assert map_torch_dtype_to_safetensors(torch.float64, None) == "F64"
    assert map_torch_dtype_to_safetensors(torch.int8, None) == "I8"
    assert map_torch_dtype_to_safetensors(torch.int16, None) == "I16"
    assert map_torch_dtype_to_safetensors(torch.int32, None) == "I32"
    assert map_torch_dtype_to_safetensors(torch.int64, None) == "I64"

    # Test with quantization config override
    mock_config = type('MockConfig', (), {
        'quantization_config': {'quant_method': 'fp8'}
    })()
    # When config has quantization, use that for low-precision dtypes
    result = map_torch_dtype_to_safetensors(torch.float16, mock_config)
    assert result == "F16"  # F16 is high precision, no override
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer
python3 -m pytest tests/capacity_planner_test.py::test_map_torch_dtype_to_safetensors -v
```

Expected: FAIL with `NameError: name 'map_torch_dtype_to_safetensors' is not defined`

**Step 3: Write minimal implementation**

Add to `capacity_planner.py` after the `precision_to_byte()` function (around line 354):

```python
def map_torch_dtype_to_safetensors(torch_dtype, model_config) -> str:
    """
    Map torch dtype to safetensors dtype string.

    Args:
        torch_dtype: A torch.dtype object (e.g., torch.float16)
        model_config: Model config to check for quantization overrides (can be None)

    Returns:
        Safetensors-compatible dtype string (e.g., "F16", "BF16")
    """
    import torch

    dtype_mapping = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
    }

    if torch_dtype in dtype_mapping:
        return dtype_mapping[torch_dtype]

    # Fallback: try to extract from dtype name
    dtype_str = str(torch_dtype).replace("torch.", "").upper()
    return dtype_str
```

**Step 4: Run test to verify it passes**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer
python3 -m pytest tests/capacity_planner_test.py::test_map_torch_dtype_to_safetensors -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add config_explorer/src/config_explorer/capacity_planner.py config_explorer/tests/capacity_planner_test.py
git commit -m "feat: add map_torch_dtype_to_safetensors helper"
```

---

## Task 3: Implement `get_model_params_from_download()` core function

**Files:**
- Modify: `config_explorer/src/config_explorer/capacity_planner.py`
- Test: `config_explorer/tests/capacity_planner_test.py`

**Step 1: Write the failing test**

Add to `capacity_planner_test.py`:

```python
def test_get_model_params_from_download():
    """Tests fallback parameter retrieval via model download"""
    # Use a small model for testing
    facebook_model = "facebook/opt-125m"

    result = get_model_params_from_download(facebook_model)

    # Should return a dict with 'parameters' and 'total' keys
    assert result is not None
    assert "parameters" in result
    assert "total" in result

    # Parameters should be a dict of dtype -> count
    assert isinstance(result["parameters"], dict)
    assert len(result["parameters"]) > 0

    # Total should be positive and match sum of parameters
    assert result["total"] > 0
    assert result["total"] == sum(result["parameters"].values())

    # For OPT-125M, we expect around 125M parameters
    assert 100_000_000 < result["total"] < 200_000_000


def test_get_model_params_from_download_invalid_model():
    """Tests fallback returns None for invalid model"""
    result = get_model_params_from_download("invalid/nonexistent-model-12345")
    assert result is None
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer
python3 -m pytest tests/capacity_planner_test.py::test_get_model_params_from_download -v
```

Expected: FAIL with `NameError: name 'get_model_params_from_download' is not defined`

**Step 3: Write minimal implementation**

Add to `capacity_planner.py` after `get_model_config_from_hf()` function (around line 185):

```python
def get_model_params_from_download(model_name: str, hf_token: str | None = None) -> dict | None:
    """
    Fallback for models without safetensors metadata.
    Uses init_empty_weights() to avoid downloading actual weights.

    Args:
        model_name: HuggingFace model name (e.g., "facebook/opt-125m")
        hf_token: Optional HuggingFace token for gated models

    Returns:
        Dict compatible with model_info.safetensors structure:
            {"parameters": {"BF16": 7000000000, ...}, "total": 7000000000}
        Returns None if model cannot be loaded.
    """
    try:
        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM

        # Fetch config first (small download)
        config = get_model_config_from_hf(model_name, hf_token)

        # Create empty model without downloading weights
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        # Count parameters by dtype
        params_by_dtype = {}
        for name, param in model.named_parameters():
            dtype_key = map_torch_dtype_to_safetensors(param.dtype, config)
            params_by_dtype[dtype_key] = params_by_dtype.get(dtype_key, 0) + param.numel()

        total = sum(params_by_dtype.values())

        return {"parameters": params_by_dtype, "total": total}

    except Exception as e:
        # Log error but return None to allow graceful handling
        print(f"Warning: Could not load model architecture for {model_name}: {e}")
        return None
```

**Step 4: Run test to verify it passes**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer
python3 -m pytest tests/capacity_planner_test.py::test_get_model_params_from_download -v
```

Expected: PASS

**Step 5: Run all capacity planner tests to ensure no regressions**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer
python3 -m pytest tests/capacity_planner_test.py -v
```

Expected: All 28+ tests PASS

**Step 6: Commit**

```bash
git add config_explorer/src/config_explorer/capacity_planner.py config_explorer/tests/capacity_planner_test.py
git commit -m "feat: add get_model_params_from_download fallback function"
```

---

## Task 4: Add helper to check if safetensors metadata is available

**Files:**
- Modify: `config_explorer/src/config_explorer/capacity_planner.py`
- Test: `config_explorer/tests/capacity_planner_test.py`

**Step 1: Write the failing test**

Add to `capacity_planner_test.py`:

```python
def test_has_safetensors_metadata():
    """Tests detection of safetensors metadata availability"""
    # Model with safetensors
    qwen_info = get_model_info_from_hf(qwen_model)
    assert has_safetensors_metadata(qwen_info) == True

    # Model without safetensors
    facebook_info = get_model_info_from_hf("facebook/opt-125m")
    assert has_safetensors_metadata(facebook_info) == False
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer
python3 -m pytest tests/capacity_planner_test.py::test_has_safetensors_metadata -v
```

Expected: FAIL with `NameError: name 'has_safetensors_metadata' is not defined`

**Step 3: Write minimal implementation**

Add to `capacity_planner.py` after `get_model_info_from_hf()` (around line 172):

```python
def has_safetensors_metadata(model_info: ModelInfo) -> bool:
    """
    Check if model has safetensors metadata available.

    Args:
        model_info: ModelInfo from HuggingFace API

    Returns:
        True if safetensors.parameters is available and non-empty
    """
    if model_info.safetensors is None:
        return False
    if not hasattr(model_info.safetensors, 'parameters'):
        return False
    if model_info.safetensors.parameters is None:
        return False
    return len(model_info.safetensors.parameters) > 0
```

**Step 4: Run test to verify it passes**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer
python3 -m pytest tests/capacity_planner_test.py::test_has_safetensors_metadata -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add config_explorer/src/config_explorer/capacity_planner.py config_explorer/tests/capacity_planner_test.py
git commit -m "feat: add has_safetensors_metadata helper"
```

---

## Task 5: Create `SafetensorsData` wrapper class for unified interface

**Files:**
- Modify: `config_explorer/src/config_explorer/capacity_planner.py`
- Test: `config_explorer/tests/capacity_planner_test.py`

**Step 1: Write the failing test**

Add to `capacity_planner_test.py`:

```python
def test_safetensors_data_wrapper():
    """Tests SafetensorsData wrapper for unified interface"""
    # Test with real safetensors data
    qwen_info = get_model_info_from_hf(qwen_model)
    wrapper = SafetensorsData.from_model_info(qwen_info)

    assert wrapper.total == qwen_info.safetensors.total
    assert wrapper.parameters == qwen_info.safetensors.parameters

    # Test with fallback dict
    fallback_dict = {"parameters": {"BF16": 1000000}, "total": 1000000}
    wrapper2 = SafetensorsData.from_fallback(fallback_dict)

    assert wrapper2.total == 1000000
    assert wrapper2.parameters == {"BF16": 1000000}
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer
python3 -m pytest tests/capacity_planner_test.py::test_safetensors_data_wrapper -v
```

Expected: FAIL with `NameError: name 'SafetensorsData' is not defined`

**Step 3: Write minimal implementation**

Add to `capacity_planner.py` after the imports (around line 43, before `AttentionType`):

```python
@dataclass
class SafetensorsData:
    """
    Unified wrapper for safetensors parameter data.
    Works with both HuggingFace ModelInfo.safetensors and fallback dicts.
    """
    parameters: dict
    total: int

    @classmethod
    def from_model_info(cls, model_info: ModelInfo) -> "SafetensorsData":
        """Create from HuggingFace ModelInfo with safetensors data"""
        return cls(
            parameters=model_info.safetensors.parameters,
            total=model_info.safetensors.total
        )

    @classmethod
    def from_fallback(cls, fallback_dict: dict) -> "SafetensorsData":
        """Create from fallback dict returned by get_model_params_from_download()"""
        return cls(
            parameters=fallback_dict["parameters"],
            total=fallback_dict["total"]
        )
```

**Step 4: Run test to verify it passes**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer
python3 -m pytest tests/capacity_planner_test.py::test_safetensors_data_wrapper -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add config_explorer/src/config_explorer/capacity_planner.py config_explorer/tests/capacity_planner_test.py
git commit -m "feat: add SafetensorsData wrapper class"
```

---

## Task 6: Update `model_memory_req()` to accept SafetensorsData

**Files:**
- Modify: `config_explorer/src/config_explorer/capacity_planner.py`
- Test: `config_explorer/tests/capacity_planner_test.py`

**Step 1: Write the failing test**

Add to `capacity_planner_test.py`:

```python
def test_model_memory_req_with_safetensors_data():
    """Tests model_memory_req works with SafetensorsData wrapper"""
    model_info = get_model_info_from_hf(qwen_model)
    model_config = get_model_config_from_hf(qwen_model)

    # Original way (should still work)
    original_result = model_memory_req(model_info, model_config)

    # New way with SafetensorsData
    safetensors_data = SafetensorsData.from_model_info(model_info)
    new_result = model_memory_req_from_safetensors(safetensors_data, model_config)

    assert original_result == new_result
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer
python3 -m pytest tests/capacity_planner_test.py::test_model_memory_req_with_safetensors_data -v
```

Expected: FAIL with `NameError: name 'model_memory_req_from_safetensors' is not defined`

**Step 3: Write minimal implementation**

Add to `capacity_planner.py` after `model_memory_req()` (around line 440):

```python
def model_memory_req_from_safetensors(safetensors_data: SafetensorsData, model_config: AutoConfig) -> float:
    """
    Calculates the GPU memory (in GiB) required for loading the model.
    Works with SafetensorsData wrapper (from either HF API or fallback download).

    Args:
        safetensors_data: SafetensorsData with parameters dict
        model_config: Model configuration for quantization info

    Returns:
        Memory requirement in GiB
    """
    model_params = safetensors_data.parameters
    memory = 0

    # Check if model is quantized
    quantization_byte = None
    if is_quantized(model_config):
        quantization_byte = get_quant_bytes(model_config)

    for precision, num_params in model_params.items():
        precision_in_byte = precision_to_byte(precision)

        # IF FP16 or FP32, keep it as so
        if precision_in_byte >= 2:
            memory += parameter_memory_req(num_params, precision)
        else:
            # Otherwise, check if model is quantized, and use that as the precision
            if quantization_byte is not None:
                memory += parameter_precision_memory_req(num_params, quantization_byte)
            else:
                memory += parameter_memory_req(num_params, precision)

    return memory
```

**Step 4: Run test to verify it passes**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer
python3 -m pytest tests/capacity_planner_test.py::test_model_memory_req_with_safetensors_data -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add config_explorer/src/config_explorer/capacity_planner.py config_explorer/tests/capacity_planner_test.py
git commit -m "feat: add model_memory_req_from_safetensors for unified interface"
```

---

## Task 7: Integrate fallback into Streamlit UI

**Files:**
- Modify: `config_explorer/Capacity_Planner.py:89-150` (model_specification function)

**Step 1: Read current implementation to understand the flow**

The `model_specification()` function currently:
1. Gets model name from text input
2. Fetches model_info via `get_model_info_from_hf()`
3. Fetches model_config via `get_model_config_from_hf()`
4. Calculates `model_memory_req()` which fails if safetensors is missing

**Step 2: Add fallback UI logic**

Update `model_specification()` in `Capacity_Planner.py`. After fetching model_info and model_config (around line 136), add the fallback check:

```python
            # Check if safetensors metadata is available
            if not has_safetensors_metadata(model_info):
                st.warning("Safetensors metadata not available for this model.")
                st.info("You can download the model architecture to get parameter info. "
                        "Note: This may take a while for large models.")

                if st.button("Download model architecture"):
                    with st.spinner("Loading model architecture (this may take a while for large models)..."):
                        fallback_params = get_model_params_from_download(selected_model, hf_token)
                        if fallback_params is not None:
                            st.session_state['fallback_params'] = fallback_params
                            st.success(f"Found {fallback_params['total']:,} parameters")
                            st.rerun()
                        else:
                            st.error("Unable to load model architecture. This model may use a custom architecture not supported by transformers.")
                            return None

                # Check if we have fallback data from previous download
                if 'fallback_params' not in st.session_state:
                    return None

                # Use fallback data
                safetensors_data = SafetensorsData.from_fallback(st.session_state['fallback_params'])
                model_gpu_memory_req = util.pretty_round(model_memory_req_from_safetensors(safetensors_data, model_config))
                st.info("Using downloaded model architecture data")
            else:
                # Normal path - safetensors available
                safetensors_data = SafetensorsData.from_model_info(model_info)
                model_gpu_memory_req = util.pretty_round(model_memory_req(model_info, model_config))
```

**Step 3: Update imports at top of file**

Add to the imports in `Capacity_Planner.py`:

```python
from src.config_explorer.capacity_planner import (
    has_safetensors_metadata,
    get_model_params_from_download,
    SafetensorsData,
    model_memory_req_from_safetensors,
)
```

Note: Since the file uses `from src.config_explorer.capacity_planner import *`, the new functions should be available automatically.

**Step 4: Test manually**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer
streamlit run Capacity_Planner.py
```

Test with:
1. `Qwen/Qwen3-0.6B` - should work normally (has safetensors)
2. `facebook/opt-125m` - should show fallback prompt

**Step 5: Commit**

```bash
git add config_explorer/Capacity_Planner.py
git commit -m "feat: add safetensors fallback UI in Streamlit"
```

---

## Task 8: Integrate fallback into CLI

**Files:**
- Modify: `config_explorer/src/config_explorer/cli.py:53-200` (plan_capacity function)

**Step 1: Add CLI arguments**

In the `main()` function where argparse is configured, add new arguments for the `plan` subcommand:

```python
# Add to plan subparser arguments
plan_parser.add_argument(
    "--fallback-download",
    action="store_true",
    default=None,
    help="Allow downloading model architecture if safetensors metadata is unavailable. May take a while for large models."
)
plan_parser.add_argument(
    "--no-fallback-download",
    action="store_true",
    help="Skip download and continue with partial info if safetensors unavailable."
)
```

**Step 2: Update plan_capacity() function**

Modify `plan_capacity()` in `cli.py` to check for safetensors and prompt/handle fallback:

```python
def plan_capacity(args):
    """Run capacity planning analysis"""

    # Get HF token from environment if available
    hf_token = os.getenv("HF_TOKEN", None)

    try:
        # Fetch model information
        print(f"Fetching model information for {args.model}...")
        model_info = get_model_info_from_hf(args.model, hf_token)
        model_config = get_model_config_from_hf(args.model, hf_token)

        # Check for safetensors metadata
        safetensors_data = None
        if not has_safetensors_metadata(model_info):
            print(f"Warning: Safetensors metadata not available for '{args.model}'.")

            # Determine whether to download
            should_download = False
            if args.fallback_download:
                should_download = True
            elif args.no_fallback_download:
                should_download = False
            else:
                # Interactive prompt
                response = input(
                    "Download model architecture to get parameters?\n"
                    "Note: This may take a while for large models.\n"
                    "[y/N]: "
                ).strip().lower()
                should_download = response == 'y'

            if should_download:
                print("Downloading model architecture...")
                fallback_params = get_model_params_from_download(args.model, hf_token)
                if fallback_params is None:
                    sys.exit("Error: Unable to load model architecture.")
                safetensors_data = SafetensorsData.from_fallback(fallback_params)
                print(f"Found {safetensors_data.total:,} parameters")
            else:
                print("Warning: Continuing with incomplete model info")
                # Cannot proceed with calculations that need parameters
                sys.exit("Error: Cannot calculate memory requirements without parameter info.")
        else:
            safetensors_data = SafetensorsData.from_model_info(model_info)

        # Rest of the function uses safetensors_data...
```

**Step 3: Update imports in cli.py**

Add to the imports:

```python
from config_explorer.capacity_planner import (
    get_model_info_from_hf,
    get_model_config_from_hf,
    has_safetensors_metadata,
    get_model_params_from_download,
    SafetensorsData,
    model_memory_req_from_safetensors,
    # ... existing imports
)
```

**Step 4: Test manually**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer

# Test with safetensors available
python3 -m config_explorer.cli plan --model Qwen/Qwen3-0.6B

# Test fallback (interactive)
python3 -m config_explorer.cli plan --model facebook/opt-125m

# Test with flag
python3 -m config_explorer.cli plan --model facebook/opt-125m --fallback-download
```

**Step 5: Commit**

```bash
git add config_explorer/src/config_explorer/cli.py
git commit -m "feat: add safetensors fallback to CLI with --fallback-download flag"
```

---

## Task 9: Integration test with facebook/opt-125m

**Files:**
- Test: `config_explorer/tests/capacity_planner_test.py`

**Step 1: Write integration test**

Add to `capacity_planner_test.py`:

```python
def test_model_memory_req_with_fallback_integration():
    """Integration test: calculate memory for model without safetensors using fallback"""
    facebook_model = "facebook/opt-125m"

    # Get model info and config
    model_info = get_model_info_from_hf(facebook_model)
    model_config = get_model_config_from_hf(facebook_model)

    # Verify safetensors is not available
    assert has_safetensors_metadata(model_info) == False

    # Use fallback
    fallback_params = get_model_params_from_download(facebook_model)
    assert fallback_params is not None

    safetensors_data = SafetensorsData.from_fallback(fallback_params)

    # Calculate memory
    memory = model_memory_req_from_safetensors(safetensors_data, model_config)

    # OPT-125M should be roughly 0.25GB in FP32 or 0.125GB in FP16
    # (125M params * 2 bytes = 250MB for FP16)
    assert 0.1 < memory < 1.0, f"Memory {memory} GB outside expected range for OPT-125M"
```

**Step 2: Run the test**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer
python3 -m pytest tests/capacity_planner_test.py::test_model_memory_req_with_fallback_integration -v
```

Expected: PASS

**Step 3: Run all tests**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer
python3 -m pytest tests/capacity_planner_test.py -v
```

Expected: All tests PASS

**Step 4: Commit**

```bash
git add config_explorer/tests/capacity_planner_test.py
git commit -m "test: add integration test for safetensors fallback"
```

---

## Task 10: Final cleanup and documentation

**Files:**
- Modify: `config_explorer/src/config_explorer/capacity_planner.py` (docstrings)

**Step 1: Update module docstring**

Update the module docstring at the top of `capacity_planner.py` to mention the fallback capability:

```python
"""
Capacity planner for LLM inference memory estimation.

This module implements memory estimation formulas for LLM inference with vLLM:
- Model weight memory requirements
- KV cache memory for different attention mechanisms (MHA, GQA, MQA, MLA)
- Activation memory during forward pass
- CUDA graph and system overhead

For models without safetensors metadata on HuggingFace, use get_model_params_from_download()
to retrieve parameter information by instantiating the model architecture without weights.

Calculates minimum GPU requirements based on model architecture, parallelism
configuration, and workload characteristics.
"""
```

**Step 2: Commit**

```bash
git add config_explorer/src/config_explorer/capacity_planner.py
git commit -m "docs: update module docstring for safetensors fallback"
```

**Step 3: Run full test suite one more time**

```bash
cd /Users/jchen/go/src/llm-d/llm-d-benchmark/.worktrees/safetensors-fallback/config_explorer
python3 -m pytest tests/capacity_planner_test.py -v
```

Expected: All tests PASS

---

## Summary

This plan implements safetensors fallback in 10 tasks:

1. Add `accelerate` dependency
2. Implement `map_torch_dtype_to_safetensors()` helper
3. Implement `get_model_params_from_download()` core function
4. Add `has_safetensors_metadata()` helper
5. Create `SafetensorsData` wrapper class
6. Add `model_memory_req_from_safetensors()` function
7. Integrate fallback into Streamlit UI
8. Integrate fallback into CLI
9. Integration test with facebook/opt-125m
10. Final cleanup and documentation

Each task follows TDD: write failing test, implement minimal code, verify test passes, commit.
