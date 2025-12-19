# from config_explorer.capacity_planner import *

from config_explorer.capacity_planner import *


model = "redhatai/nvidia-nemotron-nano-9b-v2-fp8-dynamic"

model_info = get_model_info_from_hf(model)
model_config = get_model_config_from_hf(model)
kv_cache_detail = KVCacheDetail(model_info, model_config)
print(kv_cache_detail)
# inference_d_type = inference_dtype(model_config)
# print(inference_d_type)