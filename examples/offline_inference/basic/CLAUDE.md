# Claude Code Instructions

## Code Quality Standards
- Do not use magic numbers like 450. Always make sure you use constants and document the reason for choosing that number
- Always include usage instructions in the file itself for all new files
- Prefer editing existing files to creating new ones when possible
- Follow existing code patterns and conventions

## Patent Classification System

### Project Overview
We have built a comprehensive patent classification system with multiple optimization levels and model support. The system classifies patent texts into 9 categories using few-shot learning with large language models.

### Available Models
- **Phi-3 Medium (Recommended)**: 14B parameters, MIT license, efficient performance
- **Phi-3 Mini**: 3.8B parameters, fastest option, good for testing
- **Llama-3.1-8B**: Original baseline, requires license acceptance
- **Llama-3.1-70B**: Highest accuracy, requires significant VRAM
- **Qwen2.5-7B**: Strong alternative, Apache 2.0 license

### Core Scripts

#### 1. **classify_unified.py** (Main Classification Script)
Unified script combining all optimizations with modular feature selection:

```bash
# Basic classification (backward compatible)
python examples/offline_inference/basic/classify_unified.py --num-samples 450 --few-shot-examples 3

# Enhanced algorithmic optimizations
python examples/offline_inference/basic/classify_unified.py --num-samples 450 --enhanced-prompts --advanced-sampling --chain-of-thought

# vLLM performance optimizations
python examples/offline_inference/basic/classify_unified.py --num-samples 450 --vllm-optimizations --optimal-batching --parallel-sampling

# Full power mode (all optimizations)
python examples/offline_inference/basic/classify_unified.py --experiment --enhanced-prompts --advanced-sampling --vllm-optimizations --optimal-batching --parallel-sampling --use-logit-bias

# Ray distributed processing (multi-GPU/multi-node)
python examples/offline_inference/basic/classify_unified.py --experiment --ray-distributed --num-ray-workers 4 --enhanced-prompts --vllm-optimizations

# Compare models
python examples/offline_inference/basic/classify_unified.py --model ./model_cache/llama-3.1-8b --experiment --enhanced-prompts

# Multi-LoRA Adapter Classification (domain-specific fine-tuning)
python examples/offline_inference/basic/classify_unified.py --experiment --use-lora-adapters --enhanced-prompts

# Single domain LoRA adapter
python examples/offline_inference/basic/classify_unified.py --experiment --use-lora-adapters --lora-domain chemical_materials

# LoRA ensemble mode (confidence-weighted voting)
python examples/offline_inference/basic/classify_unified.py --experiment --use-lora-adapters --lora-ensemble-mode --enhanced-prompts
```

**Features:**
- **Algorithmic Enhancements**: Enhanced prompts, advanced sampling, chain-of-thought, confidence scoring
- **vLLM Optimizations**: Optimal batching, parallel sampling, logit bias, KV cache optimization
- **Multi-LoRA Adapters**: Domain-specific fine-tuned adapters for specialized classification
- **Text Pooling**: Automatic segmentation and pooling of long patent texts (mean, max, attention-weighted strategies)
- **Tensor Parallelism**: Split single model across multiple GPUs for memory efficiency
- **Pipeline Parallelism**: Split model layers across GPUs for deeper models
- **Ray Distributed Processing**: Horizontal scaling across multiple GPUs/nodes with fault tolerance
- **Comprehensive Reporting**: Statistical analysis, confusion matrices, recommendations

#### 2. **download_model.py** (Model Management)
Smart model downloading with license checking and size warnings:

```bash
# Download recommended Phi-3 model
python examples/offline_inference/basic/download_model.py

# Download specific models
python examples/offline_inference/basic/download_model.py --model phi3-mini
python examples/offline_inference/basic/download_model.py --model llama-8b
python examples/offline_inference/basic/download_model.py --model llama-70b

# List available models
python examples/offline_inference/basic/download_model.py --list-models

# Custom models
python examples/offline_inference/basic/download_model.py --model-id microsoft/Phi-3-vision-128k-instruct
```

#### 3. **visualize_results.py** (Results Analysis)
Comprehensive visualization and analysis tool:

```bash
# Generate full visualization report
python visualize_results.py my_results.json

# Text analysis only
python visualize_results.py my_results.json --analysis-only
```

**Generates:**
- Accuracy comparison charts across few-shot configurations
- Confusion matrix heatmaps for each configuration  
- Per-class performance analysis with problem identification
- Class distribution visualization
- Detailed recommendations for improvement

#### 4. **train_lora_adapters.py** (LoRA Adapter Training)
Train domain-specific LoRA adapters for improved patent classification:

```bash
# Train all domain adapters (run once, takes several hours)
python examples/offline_inference/basic/train_lora_adapters.py --train-all-domains

# Train specific domain adapter
python examples/offline_inference/basic/train_lora_adapters.py --domain chemical_materials --output-dir ./lora_adapters

# Test mode (smaller dataset, faster training)
python examples/offline_inference/basic/train_lora_adapters.py --domain electronics_physics --test-mode --max-samples 100

# Custom output directory
python examples/offline_inference/basic/train_lora_adapters.py --train-all-domains --output-dir ./custom_lora_adapters
```

**Domain Groups:**
- **chemical_materials**: Chemistry/Metallurgy + Textiles/Paper (Classes 2,3)
- **engineering_mechanical**: Operations/Transport + Construction + Mechanical (Classes 1,4,5)
- **electronics_physics**: Physics + Electricity (Classes 6,7) 
- **life_sciences**: Human Necessities (Class 0)
- **emerging_crosscutting**: General/Cross-sectional Technology (Class 8)

#### 5. **test_lora_system.py** (LoRA System Testing)
Validate and test the multi-adapter LoRA system:

```bash
# Validate system setup and dependencies
python examples/offline_inference/basic/test_lora_system.py --validate-system

# Test basic LoRA functionality with sample patents
python examples/offline_inference/basic/test_lora_system.py --test-basic

# Test ensemble mode functionality
python examples/offline_inference/basic/test_lora_system.py --test-ensemble --lora-adapter-dir ./lora_adapters

# Test specific domain adapter
python examples/offline_inference/basic/test_lora_system.py --test-domain chemical_materials

# Complete system validation and testing
python examples/offline_inference/basic/test_lora_system.py --validate-system --test-basic
```

### Performance Expectations

| Configuration | Throughput | Accuracy | Use Case |
|---------------|------------|----------|----------|
| **Baseline** | 50-80/sec | ~35-38% | Compatibility testing |
| **Enhanced** | 40-70/sec | ~40-45% | Better prompts + sampling |
| **vLLM Optimized** | 100-200/sec | ~38-42% | High throughput |
| **Tensor Parallel** | 120-250/sec | ~38-42% | Memory-efficient multi-GPU |
| **Full Power** | 80-150/sec | ~45-55% | Best accuracy + good speed |
| **Ray Distributed** | 200-800/sec | ~45-55% | Large-scale processing |
| **LoRA Single-Adapter** | 15-30/sec | ~50-60% | Domain-specific fine-tuning |
| **LoRA Multi-Adapter** | 10-20/sec | ~55-65% | Best accuracy with routing |
| **LoRA Ensemble** | 5-15/sec | ~60-70% | Maximum accuracy with voting |

### Hardware Recommendations

#### GPU Selection for Vast.ai:
- **Budget**: RTX 4090 (~$0.20/hour, ~55 samples/sec, 24GB) - Best $/performance
- **Balanced**: A40 48GB (~$0.30/hour, ~45 samples/sec, 48GB) - Good for larger models
- **Performance**: A100 40GB (~$1.00/hour, ~75 samples/sec, 40GB) - Speed focus
- **Maximum**: A100 80GB (~$2.00/hour, ~80 samples/sec, 80GB) - Handles any model

⚠️ **Note**: Vast.ai shows misleading TFLOPS numbers (FP32 instead of AI-relevant FP16). A100 outperforms A40 for LLMs despite lower listed TFLOPS.

#### Multi-GPU Scaling:

**Tensor Parallelism (single model across GPUs):**
```bash
# 2-GPU setup (1.7-1.9x speedup)
python examples/offline_inference/basic/classify_unified.py --experiment --tensor-parallel-size 2 --vllm-optimizations

# 4-GPU setup (2.8-3.2x speedup)  
python examples/offline_inference/basic/classify_unified.py --experiment --tensor-parallel-size 4 --optimal-batching
```

**Ray Distributed (independent workers per GPU):**
```bash
# 2-GPU Ray setup (2.0x speedup, better scaling)
python examples/offline_inference/basic/classify_unified.py --experiment --ray-distributed --num-ray-workers 2

# 4-GPU Ray setup (3.5-4.0x speedup)
python examples/offline_inference/basic/classify_unified.py --experiment --ray-distributed --num-ray-workers 4 --enhanced-prompts

# 8-GPU Ray setup (6.0-8.0x speedup)
python examples/offline_inference/basic/classify_unified.py --experiment --ray-distributed --num-ray-workers 8 --ray-worker-batch-size 64

# Multi-node Ray cluster
python examples/offline_inference/basic/classify_unified.py --experiment --ray-distributed --ray-address ray://head-node:10001 --num-ray-workers 16
```

**Tensor Parallelism (single model across GPUs):**
```bash
# 2-GPU tensor parallelism (1.8-2.0x speedup, ~50% memory per GPU)
python examples/offline_inference/basic/classify_unified.py --experiment --tensor-parallel-size 2 --enhanced-prompts

# 4-GPU tensor parallelism (3.0-3.5x speedup, ~25% memory per GPU)
python examples/offline_inference/basic/classify_unified.py --experiment --tensor-parallel-size 4 --vllm-optimizations

# Pipeline parallelism (split layers across GPUs)
python examples/offline_inference/basic/classify_unified.py --experiment --pipeline-parallel-size 4 --enhanced-prompts

# Combined tensor + pipeline (for very large models)
python examples/offline_inference/basic/classify_unified.py --experiment --tensor-parallel-size 2 --pipeline-parallel-size 2
```

**Parallelism Comparison:**
- **Ray Distributed**: Best scaling, fault tolerance, works across nodes, independent workers, linear scaling
- **Tensor Parallelism**: Single model split across GPUs, reduces memory per GPU, moderate scaling
- **Pipeline Parallelism**: Splits model layers, good for very deep models, sequential processing

### Legacy Scripts (Maintained for Reference)
- **classify.py**: Original baseline implementation
- **classify_optimized.py**: Algorithmic improvements only
- **classify_vllm_optimized.py**: vLLM optimizations only

### Experimental Results Analysis
- **Current Best**: ~38% accuracy with 5-shot Llama-3.1-8B (baseline)
- **Problem Classes**: Class 8 (General/Cross-sectional: 9% F1), Classes 3,4,5 (Textiles, Construction, Mechanical: <25% F1)
- **Class Imbalance**: Severe underrepresentation in Classes 3 (0.8%) and 4 (3.1%)
- **Recommendations**: Advanced sampling, better few-shot selection, class-specific prompting

### Future Optimizations Considered
1. **Fine-tuning**: Train Phi-3 specifically on patent data
2. **Distillation**: Use larger teacher model (Llama-70B) to train efficient student
3. **BERT-style Classification**: Direct classification vs generative approach (requires architecture rewrite)
4. **Embedding + Traditional ML**: Use patent-specific embeddings with SVM/XGBoost

### Setup Requirements
- **HuggingFace Authentication**: `huggingface-cli login` (optional for Phi-3, required for Llama)
- **Dependencies**: `pip install datasets scikit-learn scipy` (for advanced features)
- **GPU**: Minimum 12GB VRAM (Phi-3), 16GB+ recommended (Llama), 24GB+ for optimal batching
- **For Llama models**: Accept license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

### Quick Start Workflow

#### Standard Classification Workflow
```bash
# 1. Download model
python examples/offline_inference/basic/download_model.py

# 2. Quick test  
python examples/offline_inference/basic/classify_unified.py --num-samples 50

# 3. Enhanced experiment
python examples/offline_inference/basic/classify_unified.py --experiment --enhanced-prompts --vllm-optimizations

# 4. Visualize results
python visualize_results.py unified_classification_report.json
```

#### Multi-LoRA Adapter Workflow (Advanced)
```bash
# 1. Validate LoRA system setup
python examples/offline_inference/basic/test_lora_system.py --validate-system

# 2. Train domain-specific LoRA adapters (takes 2-4 hours per domain)
python examples/offline_inference/basic/train_lora_adapters.py --train-all-domains

# 3. Test LoRA system with sample patents
python examples/offline_inference/basic/test_lora_system.py --test-basic

# 4. Run classification with multi-adapter routing
python examples/offline_inference/basic/classify_unified.py --experiment --use-lora-adapters --enhanced-prompts

# 5. Compare ensemble vs single-adapter modes
python examples/offline_inference/basic/classify_unified.py --experiment --use-lora-adapters --lora-ensemble-mode

# 6. Compare against baseline performance
python examples/offline_inference/basic/classify_unified.py --experiment --enhanced-prompts  # Baseline
```

#### Domain-Specific Performance Improvement Workflow
```bash
# 1. Identify problematic domain from baseline results
# (e.g., poor performance on Classes 6,7 = electronics_physics domain)

# 2. Train specific domain adapter
python examples/offline_inference/basic/train_lora_adapters.py --domain electronics_physics

# 3. Test domain-specific adapter
python examples/offline_inference/basic/classify_unified.py --experiment --use-lora-adapters --lora-domain electronics_physics

# 4. Compare domain-specific vs general classification
python examples/offline_inference/basic/classify_unified.py --experiment --enhanced-prompts  # General
python examples/offline_inference/basic/classify_unified.py --experiment --use-lora-adapters --lora-domain electronics_physics  # Specialized
```