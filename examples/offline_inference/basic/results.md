# Patent Classification System - v2: vLLM approach with larger model size for better accuracy

## System Architecture & Configuration

### Hardware Setup
- **GPUs**: 2x NVIDIA A100 80GB
- **Model**: Llama-3.1-70B-Instruct-AWQ-INT4 (quantized for efficiency)
- **GPU Memory Utilization**: 90% with 4GB swap space
- **Inference Engine**: vLLM with FlashInfer backend
- **Total Experiment Runtime**: 692.3 seconds

### Model Configuration
```python
# Final optimized configuration
MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
TENSOR_PARALLEL_SIZE = 2  # Single GPU model loading
GPU_MEMORY_UTILIZATION = 0.85
SWAP_SPACE = 4GB
BATCH_SIZE = 256 (high throughput optimization)
TEMPERATURE = 0.0 (deterministic classification)
MAX_TOKENS = 5
```

## Results
* Far better than with `accelerate`

```
=== PATENT CLASSIFICATION ANALYSIS ===

📊 OVERALL PERFORMANCE:
   • Best: 5-shot with 0.382 accuracy
   • Worst: 1-shot with 0.337 accuracy
   • Improvement: 0.045

🎯 PER-CLASS PERFORMANCE (sorted by F1 score):
   ✅ Class 2: 0.615 F1 - Chemistry; Metallurgy
   ✅ Class 0: 0.512 F1 - Human Necessities
   ⚠️ Class 6: 0.411 F1 - Physics
   ⚠️ Class 7: 0.320 F1 - Electricity
   ❌ Class 3: 0.295 F1 - Textiles; Paper
   ❌ Class 1: 0.267 F1 - Performing Operations; Transporting
   ❌ Class 4: 0.247 F1 - Fixed Constructions
   ❌ Class 5: 0.227 F1 - Mechanical Engineering; Lightning; Heating; Weapons; Blasting
   ❌ Class 8: 0.090 F1 - General tagging of new or cross-sectional technology

🚨 MAJOR PROBLEM CLASSES (F1 < 0.3):
   • Class 3 (Textiles; Paper):
     - F1: 0.295 (std: 0.037)
     - Precision: 0.218
     - Recall: 0.475
     - Sample count: 15
   • Class 1 (Performing Operations; Transporting):
     - F1: 0.267 (std: 0.037)
     - Precision: 0.279
     - Recall: 0.257
     - Sample count: 232
   • Class 4 (Fixed Constructions):
     - F1: 0.247 (std: 0.019)
     - Precision: 0.191
     - Recall: 0.362
     - Sample count: 56
   • Class 5 (Mechanical Engineering; Lightning; Heating; Weapons; Blasting):
     - F1: 0.227 (std: 0.031)
     - Precision: 0.413
     - Recall: 0.158
     - Sample count: 124
   • Class 8 (General tagging of new or cross-sectional technology):
     - F1: 0.090 (std: 0.011)
     - Precision: 0.071
     - Recall: 0.123
     - Sample count: 178

⚖️ CLASS IMBALANCE ANALYSIS:
   📊 Class 8: 10.1% (181 samples) - General tagging of new or cross-sectional technology
   📊 Class 1: 13.0% (233 samples) - Performing Operations; Transporting
   📈 Class 6: 22.2% (398 samples) - Physics
   📊 Class 2: 7.9% (141 samples) - Chemistry; Metallurgy
   📈 Class 7: 20.7% (372 samples) - Electricity
   📈 Class 0: 15.1% (271 samples) - Human Necessities
   📊 Class 5: 7.1% (128 samples) - Mechanical Engineering; Lightning; Heating; Weapons; Blasting
   📉 Class 4: 3.1% (56 samples) - Fixed Constructions
   📉 Class 3: 0.8% (15 samples) - Textiles; Paper

⚠️  SEVERELY UNDERREPRESENTED CLASSES (< 5% of data):
   • Class 4: 3.1% - Fixed Constructions
   • Class 3: 0.8% - Textiles; Paper

📋 RECOMMENDATIONS:
   • Use 5 few-shot examples for optimal performance (0.382 accuracy)
   • Classes with poor performance (F1 < 0.5): Performing Operations; Transporting, Textiles; Paper, Fixed Constructions, Mechanical Engineering; Lightning; Heating; Weapons; Blasting, Physics, Electricity, General tagging of new or cross-sectional technology
   • Consider: (1) More few-shot examples for these classes, (2) Better example selection, (3) Class-specific prompt engineering
   • For speed, use 1 examples (61.4 samples/sec, 0.337 accuracy)
```

### Results visualized

<img width="4470" height="1766" alt="image" src="https://github.com/user-attachments/assets/dc788a77-590a-4b1e-9079-62d642b74136" />

<img width="4144" height="2364" alt="image" src="https://github.com/user-attachments/assets/afa9b46f-d27a-4f68-9e11-89e5fb415704" />

<img width="5784" height="4759" alt="image" src="https://github.com/user-attachments/assets/44069174-d464-4139-a8c5-f0e4cc80c6f9" />

<img width="4517" height="2366" alt="image" src="https://github.com/user-attachments/assets/17b625ec-e5ce-41b2-8190-9171675e1917" />



## Changes and/or optimizations

1. Larger sample size from before 

2. Chain-of-Thought Reasoning

3. vLLM backend instead of accelerate with optimizations
- Use flashinfer-python==0.2.4
- 256-sample batches for maximum GPU utilization and throughput
- Multiple prediction samples per text for confidence estimation

4. Parallel Processing
- Multi-GPU Ready: Tensor parallelism: 2 
- Background Processing: Non-blocking inference for improved throughput

5. Per-class evaluations

6. Pooling

#### Top Performing Classes (F1 > 0.5)
1. **Chemistry; Metallurgy** (Class 2): F1 = 0.586
   - High recall (77.7%), indicating strong pattern recognition for chemical patents
   - Excellent keyword matching for chemical terminology
   
2. **Human Necessities** (Class 0): F1 = 0.573
   - Balanced precision/recall profile
   - Strong performance on medical, food, and consumer product patents
   
3. **Electricity** (Class 7): F1 = 0.533
   - Good recall for electrical and electronic patents
   - Clear technical vocabulary facilitates classification

#### Moderate Performance Classes (F1 0.3-0.5)
4. **Textiles; Paper** (Class 3): F1 = 0.503
5. **Fixed Constructions** (Class 4): F1 = 0.488
6. **Mechanical Engineering** (Class 5): F1 = 0.417

#### Challenging Classes (F1 < 0.4)
7. **Physics** (Class 6): F1 = 0.371
   - High precision (57.0%) but low recall (27.9%)
   - Overlap with electrical and mechanical categories
   
8. **Performing Operations; Transporting** (Class 1): F1 = 0.356
   - Broad category with significant overlap with mechanical engineering
   
9. **General/Cross-sectional Technology** (Class 8): F1 = 0.077
   - Extremely challenging due to cross-domain nature
   - Low sample quality and ambiguous classification criteria

## Dataset & Experimental Design

### Dataset Specifications
- **Source**: ccdv/patent-classification (HuggingFace)
- **Total Samples**: 1,493 patent texts
- **Class Distribution**: 9 patent categories (IPC classification system)
- **Sampling Strategy**: Advanced stratified sampling with class balance optimization

### Patent Classes Evaluated
```
0: Human Necessities (160 samples)
1: Performing Operations; Transporting (205 samples)
2: Chemistry; Metallurgy (143 samples)
3: Textiles; Paper (102 samples)
4: Fixed Constructions (109 samples)
5: Mechanical Engineering; Lightning; Heating; Weapons; Blasting (124 samples)
6: Physics (256 samples)
7: Electricity (204 samples)
8: General tagging of new or cross-sectional technology (190 samples)
```

### Confusion Matrix Insights

**Key Misclassification Patterns:**
- **Physics ↔ Electricity**: 116 physics patents misclassified as electricity
- **General Tech → Chemistry**: 42 general technology patents misclassified as chemistry
- **Mechanical → Performing Operations**: 37 mechanical engineering patents confused with operations


## Next steps I would try
1. [TF IDF + LogReg -> LLM: hierarchical classification](https://github.com/kaashmonee/vllm/blob/tfidf-hierarchical/examples/offline_inference/basic/classify_unified.py)
2. [Trained LoRA + voting per class approach](https://github.com/kaashmonee/vllm/blob/lora/examples/offline_inference/basic/classify_unified.py)
   a. unable to do this this time since I believe it's not compatible with the 4 bit quantized Llama 70b model I was using
3. Use full 16fp Llama 70b model with domain-specific LoRA and distill to smaller model. Consider doing so with ensemble methods...?
4. Use non-vLLM approach: sglang? [BELT?](https://github.com/mim-solutions/bert_for_longer_texts/blob/main/notebooks/regression/belt.ipynb) 

