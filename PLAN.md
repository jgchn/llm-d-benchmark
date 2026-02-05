# Plan for Issue #382: Obtain Parameters for Models Without Safetensor Data

## Problem Summary

Models like `facebook/opt-125m` and `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` lack safetensor metadata on HuggingFace. The current `model_memory_req()` function fails with an Exception when `model_info.safetensors` is `None`.

---

## Proposed Solution: Multi-Strategy Model Parameter Retrieval

### Strategy 1: Empty Weights Instantiation (Fast, Preferred)

Use `accelerate.init_empty_weights()` to instantiate the model architecture without downloading weights:

```
AutoConfig.from_pretrained() -> init_empty_weights() -> AutoModelForCausalLM.from_config() -> sum parameters
```

**Pros:** Only downloads `config.json` (~KB), very fast
**Cons:** Doesn't work for all models (e.g., OPT architecture lacks some config fields)

### Strategy 2: Full Model Download (Slow, Fallback)

Download the actual model weights using `AutoModelForCausalLM.from_pretrained()` and count parameters.

**Pros:** Works for all models
**Cons:** Downloads full weights (can be 100s of GB), slow

---

## API Contract Changes

### 1. New Function in `src/config_explorer/capacity_planner.py`

```python
def get_model_params_fallback(
    model_name: str,
    model_config: AutoConfig,
    hf_token: str | None = None,
    allow_download: bool = False
) -> dict[str, int] | None:
    """
    Fallback parameter retrieval for models without safetensor metadata.

    Returns:
        dict mapping dtype strings to parameter counts (matching safetensors.parameters format)
        or None if retrieval fails

    Args:
        model_name: HuggingFace model identifier
        model_config: Pre-fetched AutoConfig
        hf_token: Optional HF token for gated models
        allow_download: If True, allows full model download as last resort
    """
```

### 2. Modify `model_memory_req()` (lines 413-439)

Add fallback logic when `model_info.safetensors` is `None`:

```python
def model_memory_req(model_info, model_config, allow_download=False):
    if model_info.safetensors is None:
        params_dict = get_model_params_fallback(
            model_info.id, model_config, allow_download=allow_download
        )
        if params_dict is None:
            raise ValueError(f"Cannot determine parameters for {model_info.id}")
        # Use params_dict instead of model_info.safetensors.parameters
    else:
        # Existing logic
```

### 3. New Helper Function

```python
def infer_dtype_from_config(model_config: AutoConfig) -> str:
    """
    Infer the primary dtype from model config.
    Returns dtype string compatible with precision_to_byte().
    """
```

---

## UI Changes (`Capacity_Planner.py`)

### 1. Add Checkbox in Sidebar (around line 280)

```python
allow_model_download = st.sidebar.checkbox(
    "Allow model download for missing safetensor data",
    value=False,
    help="If enabled, will download model weights to determine parameters. This can be slow for large models."
)
```

### 2. Modify Exception Handling (lines 308-310, 516-517)

Replace broad `except Exception` with specific handling:
- First attempt: empty weights strategy
- If `allow_model_download` enabled: full download strategy
- Show appropriate warning with estimated download size

---

## CLI Changes (`cli.py`)

Add new argument to `plan_capacity`:

```python
parser.add_argument(
    "--allow-download",
    action="store_true",
    help="Allow downloading model weights if safetensor metadata unavailable"
)
```

---

## Implementation Order

1. **Add `get_model_params_fallback()`** in `capacity_planner.py` with empty weights strategy only
2. **Add `infer_dtype_from_config()`** helper
3. **Modify `model_memory_req()`** to use fallback when safetensors is None
4. **Add tests** for `facebook/opt-125m` (currently expected to fail)
5. **Update CLI** with `--allow-download` flag
6. **Update Streamlit UI** with checkbox and improved error messages
7. **(Optional)** Add full download strategy behind `allow_download=True` flag

---

## Dependencies to Add

In `requirements.txt`:
- `accelerate>=0.20.0` (for `init_empty_weights`)
- `torch>=2.0.0` (already likely present, needed for parameter counting)

---

## Test Cases to Add

```python
def test_model_without_safetensors_empty_weights():
    """Test empty weights fallback for models without safetensor data."""
    model_name = "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
    model_info = get_model_info_from_hf(model_name)
    model_config = get_model_config_from_hf(model_name)

    params = get_model_params_fallback(model_name, model_config)
    assert params is not None
    assert sum(params.values()) > 0

def test_opt_model_fallback():
    """Test fallback for facebook/opt-125m."""
    # This should work with empty weights strategy
    # or gracefully indicate download is needed
```

---

## Key Considerations

1. **Cache fallback results** - Store computed parameters in `db.json` or a separate cache to avoid re-computation
2. **Progress indicator** - For full downloads, show progress in Streamlit
3. **Disk space warning** - Alert users before large downloads
4. **Dtype inference** - When counting parameters via torch, infer dtype from the loaded model rather than assuming
