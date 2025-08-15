- do not use magic numbers like 450. always make sure you use constants and document the reason for choosing that number

## Patent Classification Experiment
### Setup Requirements
- HuggingFace authentication: `huggingface-cli login` or set `HF_TOKEN` environment variable
- Accept Llama 3.1 license: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- Install dependencies: `pip install datasets` (for loading patent dataset)

### Usage Commands
- **Full Experiment**: `python classify.py --experiment` (tests 1,2,3,5-shot configs on 1800 samples)
- **Quick Test**: `python classify.py --num_samples 450 --few_shot_examples 3`
- **Custom Report**: `python classify.py --experiment --output_report my_results.json`

### What the Experiment Tests
- Evaluates Llama 3.1 8B on patent classification (9 classes)
- Uses stratified sampling to handle class imbalance 
- Tests different few-shot configurations (1, 2, 3, 5 examples per class)
- Generates statistical analysis with 95% confidence intervals
- Outputs detailed JSON report with per-class metrics and recommendations

### Expected Results
- Takes ~30-60 minutes for full experiment (1800 samples × 4 configs)
- Produces comprehensive report with accuracy, precision, recall, F1 per class
- Identifies optimal few-shot configuration and problematic classes
- Includes confusion matrices and performance vs speed tradeoffs
- always include usage instructions in the file itself for all new files