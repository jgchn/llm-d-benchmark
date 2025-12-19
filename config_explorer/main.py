from config_explorer.recommender import recommend_gpus, ModelArchitecture

# Load model
model = ModelArchitecture("Qwen/Qwen3-32B")

# Get recommendations
recs = recommend_gpus(
    model=model,
    input_length=512,
    output_length=256,
    precision="fp16"
)

# Display results
print(recs.summary())