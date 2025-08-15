#!/usr/bin/env python3

"""
Model Download Script for Patent Classification

This script downloads models for patent classification experiments.
Supports multiple model options including Phi-3, Llama, and others.

USAGE EXAMPLES:

Download Phi-3 (default, more efficient):
    python examples/offline_inference/basic/download_model.py

Download Phi-3 Mini (smallest):
    python examples/offline_inference/basic/download_model.py --model phi3-mini

Download Llama-3.1-8B (original):
    python examples/offline_inference/basic/download_model.py --model llama-8b

Download Llama-3.1-70B (large):
    python examples/offline_inference/basic/download_model.py --model llama-70b

Download custom model:
    python examples/offline_inference/basic/download_model.py --model-id microsoft/Phi-3-medium-4k-instruct --cache-dir ./custom_cache

List available models:
    python examples/offline_inference/basic/download_model.py --list-models

REQUIREMENTS:
- huggingface_hub: pip install huggingface_hub
- HF_TOKEN environment variable (optional, for faster downloads)
- For Llama models: Accept license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
"""

from huggingface_hub import snapshot_download
import os
import argparse
from pathlib import Path

# --- Model Configurations ---
AVAILABLE_MODELS = {
    "phi3": {
        "repo_id": "microsoft/Phi-3-medium-4k-instruct",
        "local_dir": "./model_cache/phi3-medium",
        "description": "Phi-3 Medium (14B) - Efficient and capable",
        "size": "~28GB",
        "license": "MIT (no license acceptance required)",
        "recommended": True
    },
    "phi3-mini": {
        "repo_id": "microsoft/Phi-3-mini-4k-instruct", 
        "local_dir": "./model_cache/phi3-mini",
        "description": "Phi-3 Mini (3.8B) - Fastest option",
        "size": "~8GB",
        "license": "MIT (no license acceptance required)",
        "recommended": False
    },
    "phi3-small": {
        "repo_id": "microsoft/Phi-3-small-8k-instruct",
        "local_dir": "./model_cache/phi3-small", 
        "description": "Phi-3 Small (7B) - Good balance",
        "size": "~14GB",
        "license": "MIT (no license acceptance required)",
        "recommended": False
    },
    "llama-8b": {
        "repo_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "local_dir": "./model_cache/llama-3.1-8b",
        "description": "Llama-3.1 8B - Original baseline model",
        "size": "~16GB", 
        "license": "Llama 3.1 License (requires acceptance)",
        "recommended": False
    },
    "llama-70b": {
        "repo_id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "local_dir": "./model_cache/llama-3.1-70b",
        "description": "Llama-3.1 70B - Highest accuracy",
        "size": "~140GB",
        "license": "Llama 3.1 License (requires acceptance)", 
        "recommended": False
    },
    "qwen": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "local_dir": "./model_cache/qwen2.5-7b",
        "description": "Qwen2.5 7B - Strong alternative",
        "size": "~14GB",
        "license": "Apache 2.0 (no license acceptance required)",
        "recommended": False
    }
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download models for patent classification")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="phi3",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to download (default: phi3)"
    )
    
    parser.add_argument(
        "--model-id",
        type=str,
        help="Custom HuggingFace model ID (overrides --model)"
    )
    
    parser.add_argument(
        "--cache-dir", 
        type=str,
        help="Custom cache directory (overrides default)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true", 
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if model exists"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace token (defaults to HF_TOKEN env var)"
    )
    
    return parser.parse_args()

def list_available_models():
    """Display available models in a nice format."""
    print("=" * 80)
    print("AVAILABLE MODELS FOR PATENT CLASSIFICATION")
    print("=" * 80)
    
    for model_key, config in AVAILABLE_MODELS.items():
        status = "⭐ RECOMMENDED" if config.get("recommended") else "  "
        print(f"\n{status} Model: {model_key}")
        print(f"  Repository: {config['repo_id']}")
        print(f"  Description: {config['description']}")
        print(f"  Download Size: {config['size']}")
        print(f"  License: {config['license']}")
        print(f"  Cache Location: {config['local_dir']}")
    
    print(f"\n" + "=" * 80)
    print("USAGE:")
    print(f"  python {__file__} --model phi3")
    print(f"  python {__file__} --model llama-8b") 
    print(f"  python {__file__} --model-id your/custom-model")

def get_model_config(args):
    """Get model configuration based on arguments."""
    if args.model_id:
        # Custom model ID provided
        model_name = args.model_id.split("/")[-1] if "/" in args.model_id else args.model_id
        return {
            "repo_id": args.model_id,
            "local_dir": args.cache_dir or f"./model_cache/{model_name}",
            "description": f"Custom model: {args.model_id}"
        }
    else:
        # Use predefined model
        config = AVAILABLE_MODELS[args.model].copy()
        if args.cache_dir:
            config["local_dir"] = args.cache_dir
        return config

def check_license_requirements(config):
    """Check and warn about license requirements."""
    repo_id = config["repo_id"]
    
    if "llama" in repo_id.lower():
        print("⚠️  LLAMA LICENSE REQUIRED:")
        print("   1. Go to: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
        print("   2. Accept the license agreement")  
        print("   3. Ensure you're logged in: huggingface-cli login")
        print()
        
        response = input("Have you accepted the Llama license? (y/N): ")
        if response.lower() != 'y':
            print("❌ Please accept the Llama license before downloading.")
            return False
    
    return True

def download_model(config, token, force_download=False):
    """Download the model with progress tracking."""
    repo_id = config["repo_id"]
    local_dir = Path(config["local_dir"])
    
    # Check if model already exists
    if local_dir.exists() and not force_download:
        model_files = list(local_dir.glob("*.bin")) + list(local_dir.glob("*.safetensors"))
        if model_files:
            print(f"✅ Model already exists at: {local_dir}")
            print("   Use --force-download to re-download")
            return True
    
    # Create directory
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading model: {repo_id}")
    print(f"📁 Destination: {local_dir}")
    print(f"📊 Expected size: {config.get('size', 'Unknown')}")
    print()
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # Recommended for stability
            token=token,
            resume_download=True,  # Resume interrupted downloads
        )
        
        print(f"\n✅ Download complete!")
        print(f"📁 Model saved to: {local_dir}")
        
        # Show how to use the model
        print(f"\n🚀 USAGE:")
        print(f"python examples/offline_inference/basic/classify_unified.py --model {local_dir} --num-samples 450")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Check internet connection")
        print(f"2. Verify HuggingFace token: huggingface-cli login")
        print(f"3. For Llama models, ensure license is accepted")
        print(f"4. Try again with --force-download")
        return False

def main():
    """Main function."""
    args = parse_args()
    
    # List models and exit
    if args.list_models:
        list_available_models()
        return
    
    # Get model configuration
    config = get_model_config(args)
    
    # Get token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("💡 TIP: Set HF_TOKEN environment variable for faster downloads")
        print("   Run: huggingface-cli login")
        print()
    
    # Check license requirements
    if not check_license_requirements(config):
        return
    
    # Show model info
    print("=" * 80)
    print("MODEL DOWNLOAD")
    print("=" * 80)
    print(f"Model: {config['repo_id']}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Cache Location: {config['local_dir']}")
    print("=" * 80)
    print()
    
    # Download model
    success = download_model(config, token, args.force_download)
    
    if success:
        print(f"\n🎉 Ready for patent classification!")
        print(f"Next steps:")
        print(f"1. Test: python examples/offline_inference/basic/classify_unified.py --num-samples 50")
        print(f"2. Experiment: python examples/offline_inference/basic/classify_unified.py --experiment --enhanced-prompts")
    else:
        exit(1)

if __name__ == "__main__":
    main()