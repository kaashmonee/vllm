from huggingface_hub import snapshot_download
import os

# --- Configuration ---
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
local_model_dir = "./model_cache/llama-3.1-8b"
token = os.environ.get("HF_TOKEN") # Make sure your HF_TOKEN is set as an environment variable

# --- Download the model ---
print(f"Downloading model '{model_name}' to '{local_model_dir}'...")

snapshot_download(
    repo_id=model_name,
    local_dir=local_model_dir,
    local_dir_use_symlinks=False,  # Recommended for stability
    token=token,
    resume_download=True, # Resume interrupted downloads
)

print("\n✅ Download complete!")
print(f"Model is saved in: {local_model_dir}")