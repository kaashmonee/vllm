#!/usr/bin/env python3

"""
LoRA Adapter Training for Patent Classification

This script trains domain-specific LoRA adapters for patent classification
using the HuggingFace PEFT library with optimized configurations for each domain.

SETUP REQUIREMENTS:
1. Install PEFT: pip install peft accelerate bitsandbytes
2. HuggingFace authentication: huggingface-cli login
3. Sufficient VRAM: 24GB+ recommended for Llama-3.1-70B training
4. Dataset: ccdv/patent-classification from HuggingFace

USAGE:
# Train all domain adapters
python train_lora_adapters.py --train-all-domains

# Train specific domain
python train_lora_adapters.py --domain chemical_materials --output-dir ./lora_adapters

# Resume training
python train_lora_adapters.py --domain electronics_physics --resume-from ./lora_adapters/electronics_physics_checkpoint

# Quick test training (smaller dataset)
python train_lora_adapters.py --domain life_sciences --test-mode --max-samples 100
"""

import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
import accelerate
from collections import Counter

# Import our domain configuration
from lora_domain_config import (
    PATENT_DOMAIN_GROUPS, 
    LORA_TRAINING_CONFIGS, 
    get_domain_for_class,
    get_classes_for_domain
)

# Constants
BASE_MODEL_NAME = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
RANDOM_SEED = 42
MAX_SEQUENCE_LENGTH = 2048
PADDING_TOKEN = "<pad>"

class PatentLoRATrainer:
    """Trainer class for domain-specific LoRA adapters."""
    
    def __init__(self, base_model_name: str = BASE_MODEL_NAME, device_map: str = "auto"):
        self.base_model_name = base_model_name
        self.device_map = device_map
        self.tokenizer = None
        self.base_model = None
        self.current_domain = None
        
        # Set random seed for reproducibility
        set_seed(RANDOM_SEED)
        
    def initialize_base_model(self):
        """Initialize the base model and tokenizer."""
        print(f"🚀 Loading base model: {self.base_model_name}")
        start_time = time.time()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": PADDING_TOKEN})
        
        # Load model - AWQ models are already quantized, others need 4-bit quantization
        model_loading_kwargs = {
            "device_map": self.device_map,
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
        }
        
        # Only add quantization for non-AWQ models
        if "AWQ" not in self.base_model_name.upper():
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_loading_kwargs["quantization_config"] = quantization_config
            print("🔧 Using BitsAndBytesConfig for 4-bit quantization")
        else:
            print("🔧 Using pre-quantized AWQ model")
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_loading_kwargs
        )
        
        # Resize embeddings if we added tokens
        if self.tokenizer.pad_token == PADDING_TOKEN:
            self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Check for incompatible model configurations
        if "AWQ" in self.base_model_name.upper():
            raise ValueError(
                f"❌ INCOMPATIBLE MODEL: {self.base_model_name}\n"
                f"AWQ quantized models are NOT compatible with LoRA training!\n"
                f"Use a non-quantized model instead:\n"
                f"  - meta-llama/Meta-Llama-3.1-8B-Instruct\n"
                f"  - meta-llama/Meta-Llama-3.1-70B-Instruct\n"
                f"  - microsoft/Phi-3-medium-4k-instruct"
            )
        
        # Prepare model for k-bit training
        try:
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        except Exception as e:
            raise RuntimeError(f"❌ FAILED to prepare model for training: {e}")
        
        # Ensure model supports gradient computation
        self.base_model.train()
        
        # Verify the model can be trained
        test_params = list(self.base_model.parameters())
        if not test_params:
            raise RuntimeError("❌ Model has no parameters - cannot train!")
        
        # Test that at least some parameters can have gradients
        trainable_found = False
        for param in test_params:
            if param.requires_grad:
                trainable_found = True
                break
        
        if not trainable_found:
            raise RuntimeError("❌ No trainable parameters found in base model!")
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        print(f"📊 Model parameters: {self.base_model.num_parameters():,}")
        
    def load_domain_dataset(self, domain: str, max_samples_per_class: Optional[int] = None) -> Dataset:
        """Load and prepare training dataset for specific domain."""
        print(f"📚 Loading dataset for domain: {domain}")
        
        # Get domain configuration
        domain_config = PATENT_DOMAIN_GROUPS[domain]
        target_classes = domain_config["classes"]
        
        # Load full patent dataset
        dataset = load_dataset("ccdv/patent-classification", split="train")
        
        # Filter for domain classes
        domain_samples = {"text": [], "label": []}
        class_counts = Counter()
        
        for text, label in zip(dataset["text"], dataset["label"]):
            if label in target_classes:
                if max_samples_per_class is None or class_counts[label] < max_samples_per_class:
                    domain_samples["text"].append(text)
                    domain_samples["label"].append(label)
                    class_counts[label] += 1
        
        print(f"📊 Domain dataset loaded:")
        print(f"   Total samples: {len(domain_samples['text'])}")
        print(f"   Class distribution: {dict(class_counts)}")
        
        # Create HuggingFace dataset
        return Dataset.from_dict(domain_samples)
    
    def create_training_prompts(self, dataset: Dataset, domain: str) -> Dataset:
        """Create training prompts for the domain dataset."""
        print(f"📝 Creating training prompts for domain: {domain}")
        
        domain_config = PATENT_DOMAIN_GROUPS[domain]
        class_names = domain_config["target_classes"]
        
        def format_training_example(example):
            """Format a single example as a training prompt."""
            text = example["text"][:800]  # Truncate long texts
            label = example["label"]
            class_name = class_names[label]
            
            # Create enhanced prompt similar to classify_unified.py
            prompt = f"""You are a patent classification expert specializing in {domain_config['description']}.

Classify this patent text into the appropriate category:

{', '.join([f"{k}: {v}" for k, v in class_names.items()])}

Key domain terms: {', '.join(domain_config['keywords'][:10])}

Patent Text: {text}

Classification: {label}"""
            
            return {"text": prompt}
        
        # Apply formatting
        formatted_dataset = dataset.map(format_training_example, remove_columns=dataset.column_names)
        
        print(f"✅ Created {len(formatted_dataset)} training prompts")
        return formatted_dataset
    
    def create_lora_config(self, domain: str) -> LoraConfig:
        """Create LoRA configuration for specific domain."""
        config = LORA_TRAINING_CONFIGS[domain]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["rank"],
            lora_alpha=config["alpha"],
            lora_dropout=config["dropout"],
            target_modules=config["target_modules"],
            bias="none",
        )
        
        print(f"🔧 LoRA configuration for {domain}:")
        print(f"   Rank: {config['rank']}")
        print(f"   Alpha: {config['alpha']}")
        print(f"   Dropout: {config['dropout']}")
        print(f"   Target modules: {config['target_modules']}")
        
        return lora_config
    
    def train_domain_adapter(self, domain: str, output_dir: str, test_mode: bool = False):
        """Train LoRA adapter for specific domain."""
        print(f"🎯 Training LoRA adapter for domain: {domain}")
        start_time = time.time()
        
        # Create output directory
        domain_output_dir = Path(output_dir) / domain
        domain_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load domain dataset
        training_config = LORA_TRAINING_CONFIGS[domain]
        max_samples = 100 if test_mode else training_config["training_samples_per_class"]
        dataset = self.load_domain_dataset(domain, max_samples)
        
        # Create training prompts
        train_dataset = self.create_training_prompts(dataset, domain)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding=False,
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors=None
            )
        
        tokenized_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        # Create LoRA model
        lora_config = self.create_lora_config(domain)
        model = get_peft_model(self.base_model, lora_config)
        
        # Enable training mode and ensure gradients are enabled
        model.train()
        
        # Enable gradients for LoRA parameters
        for name, param in model.named_parameters():
            if "lora_" in name or "modules_to_save" in name:
                param.requires_grad = True
        
        print(f"📊 LoRA model statistics:")
        model.print_trainable_parameters()
        
        # Verify gradients are properly set
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🔍 Verified trainable parameters: {trainable_params:,}")
        
        if trainable_params == 0:
            raise ValueError("No parameters have requires_grad=True! LoRA setup failed.")
        
        # Training arguments
        epochs = 2 if test_mode else training_config["epochs"]
        training_args = TrainingArguments(
            output_dir=str(domain_output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=1,  # Small batch size for large model
            gradient_accumulation_steps=8,  # Effective batch size = 8
            learning_rate=training_config["learning_rate"],
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=True,  # Use mixed precision
            gradient_checkpointing=True,  # Save memory
            report_to=None,  # Disable wandb
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal language modeling
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        print(f"🏋️ Starting training for {epochs} epochs...")
        trainer.train()
        
        # Save the final adapter
        final_output_dir = domain_output_dir / "final"
        trainer.save_model(str(final_output_dir))
        
        # Save domain metadata
        metadata = {
            "domain": domain,
            "training_config": training_config,
            "dataset_size": len(train_dataset),
            "classes": PATENT_DOMAIN_GROUPS[domain]["classes"],
            "target_classes": PATENT_DOMAIN_GROUPS[domain]["target_classes"],
            "training_time_seconds": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "base_model": self.base_model_name,
            "lora_config": {
                "rank": lora_config.r,
                "alpha": lora_config.lora_alpha,
                "dropout": lora_config.lora_dropout,
                "target_modules": lora_config.target_modules,
            }
        }
        
        with open(final_output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds")
        print(f"💾 Adapter saved to: {final_output_dir}")
        
        return str(final_output_dir)

def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapters for patent classification")
    parser.add_argument("--domain", type=str, choices=list(PATENT_DOMAIN_GROUPS.keys()),
                       help="Domain to train (if not specified with --train-all-domains)")
    parser.add_argument("--train-all-domains", action="store_true",
                       help="Train adapters for all domains")
    parser.add_argument("--output-dir", type=str, default="./lora_adapters",
                       help="Output directory for trained adapters")
    parser.add_argument("--test-mode", action="store_true",
                       help="Quick test mode with reduced dataset size and epochs")
    parser.add_argument("--device-map", type=str, default="auto",
                       help="Device mapping strategy")
    parser.add_argument("--resume-from", type=str,
                       help="Resume training from checkpoint directory")
    
    args = parser.parse_args()
    
    if not args.domain and not args.train_all_domains:
        parser.error("Must specify either --domain or --train-all-domains")
    
    # Initialize trainer
    print("🚀 Initializing Patent LoRA Trainer")
    trainer = PatentLoRATrainer(device_map=args.device_map)
    trainer.initialize_base_model()
    
    # Determine domains to train
    if args.train_all_domains:
        domains_to_train = list(PATENT_DOMAIN_GROUPS.keys())
    else:
        domains_to_train = [args.domain]
    
    print(f"📋 Training domains: {domains_to_train}")
    
    # Train adapters
    trained_adapters = {}
    total_start_time = time.time()
    
    for domain in domains_to_train:
        try:
            print(f"\n{'='*60}")
            print(f"Training domain: {domain}")
            print(f"{'='*60}")
            
            adapter_path = trainer.train_domain_adapter(
                domain=domain,
                output_dir=args.output_dir,
                test_mode=args.test_mode
            )
            trained_adapters[domain] = adapter_path
            
        except Exception as e:
            print(f"❌ Failed to train adapter for {domain}: {e}")
            continue
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total training time: {total_time:.1f} seconds")
    print(f"Successfully trained adapters: {len(trained_adapters)}")
    
    for domain, path in trained_adapters.items():
        print(f"  {domain}: {path}")
    
    if trained_adapters:
        print(f"\n🎉 Training completed! Adapters saved to: {args.output_dir}")
        print("Next steps:")
        print("1. Test the adapters with classify_unified.py --use-lora-adapters")
        print("2. Compare performance against baseline model")
        print("3. Fine-tune adapter configurations if needed")
    else:
        print("❌ No adapters were successfully trained")

if __name__ == "__main__":
    main()