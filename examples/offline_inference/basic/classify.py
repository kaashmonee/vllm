# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Patent Classification with Few-Shot Prompting

This script performs patent text classification using few-shot prompting with LLMs.
It supports both quick testing and comprehensive experiments across multiple configurations.

SETUP REQUIREMENTS:
1. HuggingFace Authentication (for Llama 3.1):
   - Run: huggingface-cli login
   - Or set: export HF_TOKEN="your_token_here"  
   - Accept license: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

2. Install Dependencies:
   - pip install datasets (for loading patent dataset)
   - pip install scipy (optional, for better confidence intervals)

USAGE EXAMPLES:

Quick Test (450 samples, 3-shot):
    python examples/offline_inference/basic/classify.py --num-samples 450 --few-shot-examples 3

Full Experiment (1800 samples, test 1,2,3,5-shot configs):
    python examples/offline_inference/basic/classify.py --experiment

Custom Experiment with Report:
    python examples/offline_inference/basic/classify.py --experiment --output-report my_results.json

Different Model:
    python examples/offline_inference/basic/classify.py --model microsoft/DialoGPT-large --num-samples 200

WHAT IT DOES:
- Loads patent classification dataset (9 classes) with stratified sampling
- Uses few-shot prompting with examples from each class
- Evaluates performance with detailed per-class metrics
- In experiment mode: tests multiple few-shot configurations
- Generates comprehensive reports with statistical analysis

EXPECTED RUNTIME:
- Quick test: ~5-10 minutes
- Full experiment: ~30-60 minutes (1800 samples × 4 configs)
"""

from argparse import Namespace
import numpy as np
from collections import Counter
import time
import json
from datetime import datetime

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser

# Import datasets library for loading patent classification data
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Add custom arguments for patent classification
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=DEFAULT_TOTAL_SAMPLES, 
        help=f"Total number of samples to classify from the dataset (default: {DEFAULT_TOTAL_SAMPLES}, ~{DEFAULT_SAMPLES_PER_CLASS} per class)"
    )
    parser.add_argument(
        "--few-shot-examples", 
        type=int, 
        default=DEFAULT_FEW_SHOT_EXAMPLES, 
        help=f"Number of examples per class for few-shot prompting (default: {DEFAULT_FEW_SHOT_EXAMPLES})"
    )
    parser.add_argument(
        "--experiment", 
        action="store_true", 
        help="Run full experiment with multiple few-shot configurations and generate detailed report"
    )
    parser.add_argument(
        "--output-report", 
        type=str, 
        default="patent_classification_report.json", 
        help="Output file for experiment report (JSON format)"
    )
    # Set example specific arguments - using a more powerful model for few-shot classification
    # NOTE: Llama models require HuggingFace authentication
    # Run: huggingface-cli login or set HF_TOKEN environment variable
    # Accept license at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
    parser.set_defaults(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",  # Larger model for better few-shot prompting
        enforce_eager=True,
    )
    return parser.parse_args()


# Patent classification class labels (9 classes total)
PATENT_CLASSES = {
    0: "Human Necessities",
    1: "Performing Operations; Transporting", 
    2: "Chemistry; Metallurgy",
    3: "Textiles; Paper",
    4: "Fixed Constructions",
    5: "Mechanical Engineering; Lightning; Heating; Weapons; Blasting",
    6: "Physics", 
    7: "Electricity",
    8: "General tagging of new or cross-sectional technology"
}

# Configuration constants
NUM_PATENT_CLASSES = 9

# Default sample size calculation:
# - Minimum 20 samples per class needed for meaningful statistical evaluation
# - With 9 classes, that's 20 * 9 = 180 minimum samples
# - Adding buffer for class imbalance (smallest class gets ~4% of samples)
# - 180 / 0.04 = 4500, but that's too large for demo purposes
# - Compromise: 50 samples per class average = 450 total samples
# - This ensures minority classes (4% of dataset) get ~18 samples minimum
MIN_SAMPLES_PER_CLASS = 20
DEFAULT_SAMPLES_PER_CLASS = 50
DEFAULT_TOTAL_SAMPLES = DEFAULT_SAMPLES_PER_CLASS * NUM_PATENT_CLASSES

# Experiment configuration
# Large-scale evaluation with 200 samples per class for robust statistics
EXPERIMENT_SAMPLES_PER_CLASS = 200
EXPERIMENT_TOTAL_SAMPLES = EXPERIMENT_SAMPLES_PER_CLASS * NUM_PATENT_CLASSES
EXPERIMENT_FEW_SHOT_CONFIGS = [1, 2, 3, 5]  # Different few-shot configurations to test

# Few-shot prompting configuration
# Using 2 examples per class provides good context without making prompts too long
DEFAULT_FEW_SHOT_EXAMPLES = 2

# Text processing limits
# Patent texts can be very long (10k+ chars), truncate for prompt efficiency
MAX_TEXT_LENGTH_FOR_CLASSIFICATION = 500
MAX_TEXT_LENGTH_FOR_EXAMPLES = 200
MAX_TEXT_LENGTH_FOR_DISPLAY = 150

# Model generation parameters
CLASSIFICATION_TEMPERATURE = 0.1  # Low temperature for consistent classification
MAX_CLASSIFICATION_TOKENS = 10    # Only need a few tokens for class prediction
RESPONSE_PARSE_WINDOW = 10        # Look for class numbers in first 10 chars of response

# Report configuration
CONFIDENCE_LEVEL = 0.95  # 95% confidence intervals
RANDOM_SEED = 42         # For reproducible results


def stratified_sample_dataset(dataset, num_samples=DEFAULT_TOTAL_SAMPLES):
    """Sample dataset maintaining class distribution proportions."""
    # Calculate class distribution in the full dataset
    labels = dataset["label"]
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Calculate samples per class maintaining proportions
    samples_per_class = {}
    for class_id, count in class_counts.items():
        proportion = count / total_samples
        samples_for_class = max(1, int(num_samples * proportion))  # At least 1 sample per class
        samples_per_class[class_id] = samples_for_class
    
    print(f"Target samples per class: {samples_per_class}")
    
    # Sample from each class
    sampled_texts = []
    sampled_labels = []
    
    for class_id, target_count in samples_per_class.items():
        # Filter dataset for this class
        class_indices = [i for i, label in enumerate(labels) if label == class_id]
        
        # Sample from this class
        selected_indices = np.random.choice(
            class_indices, 
            size=min(target_count, len(class_indices)), 
            replace=False
        )
        
        for idx in selected_indices:
            sampled_texts.append(dataset["text"][idx])
            sampled_labels.append(dataset["label"][idx])
    
    return sampled_texts, sampled_labels


def load_few_shot_examples(num_examples_per_class=DEFAULT_FEW_SHOT_EXAMPLES):
    """Load few-shot examples for prompting."""
    if not DATASETS_AVAILABLE:
        return {}
    
    try:
        # Load training data for few-shot examples
        train_dataset = load_dataset("ccdv/patent-classification", split="train")
        
        examples = {}
        for class_id in range(NUM_PATENT_CLASSES):
            # Find examples of this class
            class_examples = [
                (text, label) for text, label in zip(train_dataset["text"], train_dataset["label"]) 
                if label == class_id
            ]
            
            # Sample a few examples
            if len(class_examples) >= num_examples_per_class:
                selected = np.random.choice(len(class_examples), num_examples_per_class, replace=False)
                examples[class_id] = [class_examples[i][0][:MAX_TEXT_LENGTH_FOR_EXAMPLES] for i in selected]
            
        return examples
        
    except Exception as e:
        print(f"Error loading few-shot examples: {e}")
        return {}


def load_patent_dataset(num_samples=DEFAULT_TOTAL_SAMPLES):
    """Load patent classification dataset with stratified sampling."""
    if not DATASETS_AVAILABLE:
        print("Using fallback sample data since datasets library is not available.")
        # Fallback sample data for demonstration
        return [
            "A method for manufacturing semiconductor devices with improved efficiency...",
            "A pharmaceutical composition comprising active compounds for treating cancer...", 
            "An apparatus for wireless communication using advanced antenna arrays...",
            "A mechanical system for automotive brake control with enhanced safety...",
            "A chemical process for producing renewable energy from biomass materials..."
        ], [6, 0, 7, 5, 2]  # Example labels
    
    try:
        # Load the patent classification dataset
        dataset = load_dataset("ccdv/patent-classification", split="test")
        
        # Use stratified sampling
        texts, labels = stratified_sample_dataset(dataset, num_samples)
        
        print(f"Loaded {len(texts)} samples from patent classification dataset")
        print(f"Class distribution: {Counter(labels)}")
        
        return texts, labels
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using fallback sample data.")
        return [
            "A method for manufacturing semiconductor devices with improved efficiency...",
            "A pharmaceutical composition comprising active compounds for treating cancer...", 
            "An apparatus for wireless communication using advanced antenna arrays...",
            "A mechanical system for automotive brake control with enhanced safety...",
            "A chemical process for producing renewable energy from biomass materials..."
        ], [6, 0, 7, 5, 2]


def create_few_shot_prompt(text_to_classify, few_shot_examples):
    """Create a few-shot prompt for patent classification."""
    
    # Create the prompt with examples
    prompt = "Classify the following patent text into one of these categories:\n\n"
    
    # Add class descriptions
    for class_id, class_name in PATENT_CLASSES.items():
        prompt += f"{class_id}: {class_name}\n"
    
    prompt += "\nHere are some examples:\n\n"
    
    # Add few-shot examples
    for class_id, examples in few_shot_examples.items():
        class_name = PATENT_CLASSES[class_id]
        for example in examples:
            prompt += f"Text: {example[:MAX_TEXT_LENGTH_FOR_EXAMPLES]}...\nClass: {class_id} ({class_name})\n\n"
    
    # Add the text to classify
    prompt += f"Now classify this text:\nText: {text_to_classify[:MAX_TEXT_LENGTH_FOR_CLASSIFICATION]}...\nClass:"
    
    return prompt


def extract_prediction_from_response(response_text):
    """Extract class prediction from model response."""
    # Look for class numbers in the response
    for i in range(NUM_PATENT_CLASSES):
        if str(i) in response_text[:RESPONSE_PARSE_WINDOW]:
            return i
    
    # Fallback: look for class names
    response_lower = response_text.lower()
    for class_id, class_name in PATENT_CLASSES.items():
        if class_name.lower()[:20] in response_lower:
            return class_id
    
    return None  # Could not extract prediction


def calculate_per_class_metrics(true_labels, predicted_labels):
    """Calculate accuracy, precision, recall per class."""
    metrics = {}
    
    for class_id in range(NUM_PATENT_CLASSES):
        # True positives, false positives, false negatives
        tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == class_id and p == class_id)
        fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t != class_id and p == class_id)
        fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == class_id and p != class_id)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        support = sum(1 for t in true_labels if t == class_id)
        
        metrics[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    return metrics


def calculate_confusion_matrix(true_labels, predicted_labels):
    """Calculate confusion matrix for multiclass classification."""
    confusion_matrix = np.zeros((NUM_PATENT_CLASSES, NUM_PATENT_CLASSES), dtype=int)
    
    for true_label, pred_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label][pred_label] += 1
    
    return confusion_matrix


def calculate_confidence_intervals(accuracy_scores, confidence_level=CONFIDENCE_LEVEL):
    """Calculate confidence intervals for accuracy scores."""
    try:
        # Use scipy if available for more accurate confidence intervals
        from scipy import stats
        mean_accuracy = np.mean(accuracy_scores)
        sem = stats.sem(accuracy_scores)  # Standard error of mean
        interval = sem * stats.t.ppf((1 + confidence_level) / 2, len(accuracy_scores) - 1)
        return mean_accuracy, mean_accuracy - interval, mean_accuracy + interval
    except ImportError:
        # Fallback to normal approximation
        mean_accuracy = np.mean(accuracy_scores)
        std_error = np.std(accuracy_scores) / np.sqrt(len(accuracy_scores))
        # Use 1.96 for 95% confidence interval (normal approximation)
        margin = 1.96 * std_error
        return mean_accuracy, mean_accuracy - margin, mean_accuracy + margin


def run_single_experiment(llm, texts, true_labels, few_shot_examples_dict, num_few_shot):
    """Run classification experiment with specified few-shot configuration."""
    print(f"\n--- Running experiment with {num_few_shot} few-shot examples ---")
    
    start_time = time.time()
    
    # Load few-shot examples for this configuration
    few_shot_subset = {}
    for class_id, examples in few_shot_examples_dict.items():
        few_shot_subset[class_id] = examples[:num_few_shot] if len(examples) >= num_few_shot else examples
    
    # Generate prompts
    prompts = []
    for text in texts:
        prompt = create_few_shot_prompt(text, few_shot_subset)
        prompts.append(prompt)
    
    # Run classification
    sampling_params = SamplingParams(temperature=CLASSIFICATION_TEMPERATURE, max_tokens=MAX_CLASSIFICATION_TOKENS)
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    
    # Extract predictions
    predictions = []
    for output in outputs:
        response = output.outputs[0].text.strip()
        prediction = extract_prediction_from_response(response)
        predictions.append(prediction)
    
    # Filter valid predictions
    valid_indices = [i for i, p in enumerate(predictions) if p is not None]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_true_labels = [true_labels[i] for i in valid_indices]
    
    end_time = time.time()
    
    # Calculate metrics
    correct = sum(1 for t, p in zip(valid_true_labels, valid_predictions) if t == p)
    overall_accuracy = correct / len(valid_predictions) if valid_predictions else 0
    per_class_metrics = calculate_per_class_metrics(valid_true_labels, valid_predictions)
    confusion_matrix = calculate_confusion_matrix(valid_true_labels, valid_predictions)
    
    return {
        'few_shot_count': num_few_shot,
        'total_samples': len(texts),
        'valid_predictions': len(valid_predictions),
        'invalid_predictions': len(predictions) - len(valid_predictions),
        'overall_accuracy': overall_accuracy,
        'correct_predictions': correct,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': confusion_matrix.tolist(),
        'processing_time_seconds': end_time - start_time,
        'samples_per_second': len(texts) / (end_time - start_time)
    }


def generate_experiment_report(experiment_results, dataset_info, model_name):
    """Generate comprehensive experiment report."""
    report = {
        'experiment_metadata': {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'dataset': 'ccdv/patent-classification',
            'total_classes': NUM_PATENT_CLASSES,
            'class_names': PATENT_CLASSES,
            'dataset_info': dataset_info,
            'experiment_config': {
                'few_shot_configs_tested': EXPERIMENT_FEW_SHOT_CONFIGS,
                'samples_per_class_target': EXPERIMENT_SAMPLES_PER_CLASS,
                'total_samples_target': EXPERIMENT_TOTAL_SAMPLES,
                'confidence_level': CONFIDENCE_LEVEL,
                'random_seed': RANDOM_SEED
            }
        },
        'results_summary': {},
        'detailed_results': experiment_results,
        'analysis': {}
    }
    
    # Summary statistics
    accuracies = [result['overall_accuracy'] for result in experiment_results]
    best_config = max(experiment_results, key=lambda x: x['overall_accuracy'])
    worst_config = min(experiment_results, key=lambda x: x['overall_accuracy'])
    
    mean_acc, ci_lower, ci_upper = calculate_confidence_intervals(accuracies)
    
    report['results_summary'] = {
        'best_few_shot_config': {
            'few_shot_count': best_config['few_shot_count'],
            'accuracy': best_config['overall_accuracy'],
            'processing_time': best_config['processing_time_seconds']
        },
        'worst_few_shot_config': {
            'few_shot_count': worst_config['few_shot_count'], 
            'accuracy': worst_config['overall_accuracy']
        },
        'overall_statistics': {
            'mean_accuracy': mean_acc,
            'accuracy_std': np.std(accuracies),
            'confidence_interval_95': [ci_lower, ci_upper],
            'accuracy_range': [min(accuracies), max(accuracies)]
        }
    }
    
    # Per-class analysis across all configurations
    class_performance = {}
    for class_id in range(NUM_PATENT_CLASSES):
        class_name = PATENT_CLASSES[class_id]
        f1_scores = []
        precisions = []
        recalls = []
        supports = []
        
        for result in experiment_results:
            metrics = result['per_class_metrics'][class_id]
            f1_scores.append(metrics['f1'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            supports.append(metrics['support'])
        
        class_performance[class_id] = {
            'class_name': class_name,
            'mean_f1': np.mean(f1_scores),
            'mean_precision': np.mean(precisions),
            'mean_recall': np.mean(recalls),
            'mean_support': np.mean(supports),
            'f1_std': np.std(f1_scores),
            'best_f1': max(f1_scores),
            'worst_f1': min(f1_scores)
        }
    
    report['analysis'] = {
        'per_class_performance': class_performance,
        'recommendations': generate_recommendations(experiment_results, class_performance)
    }
    
    return report


def generate_recommendations(experiment_results, class_performance):
    """Generate recommendations based on experiment results."""
    recommendations = []
    
    # Best few-shot configuration
    best_result = max(experiment_results, key=lambda x: x['overall_accuracy'])
    recommendations.append(f"Use {best_result['few_shot_count']} few-shot examples for optimal performance ({best_result['overall_accuracy']:.3f} accuracy)")
    
    # Identify problematic classes
    poor_classes = [(class_id, metrics) for class_id, metrics in class_performance.items() 
                   if metrics['mean_f1'] < 0.5]
    
    if poor_classes:
        poor_class_names = [metrics['class_name'] for _, metrics in poor_classes]
        recommendations.append(f"Classes with poor performance (F1 < 0.5): {', '.join(poor_class_names)}")
        recommendations.append("Consider: (1) More few-shot examples for these classes, (2) Better example selection, (3) Class-specific prompt engineering")
    
    # Performance vs speed tradeoff
    fastest = min(experiment_results, key=lambda x: x['processing_time_seconds'])
    if fastest != best_result:
        recommendations.append(f"For speed, use {fastest['few_shot_count']} examples ({fastest['samples_per_second']:.1f} samples/sec, {fastest['overall_accuracy']:.3f} accuracy)")
    
    return recommendations


def run_full_experiment(args):
    """Run comprehensive experiment with multiple few-shot configurations."""
    print("=" * 80)
    print("PATENT CLASSIFICATION EXPERIMENT")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Load large evaluation dataset
    print(f"Loading large evaluation dataset ({EXPERIMENT_TOTAL_SAMPLES} samples)...")
    texts, true_labels = load_patent_dataset(EXPERIMENT_TOTAL_SAMPLES)
    
    dataset_info = {
        'total_samples': len(texts),
        'class_distribution': dict(Counter(true_labels)),
        'sampling_strategy': 'stratified'
    }
    
    # Load few-shot examples (use maximum needed)
    max_few_shot = max(EXPERIMENT_FEW_SHOT_CONFIGS)
    print(f"Loading few-shot examples (up to {max_few_shot} per class)...")
    few_shot_examples = load_few_shot_examples(max_few_shot)
    
    # Initialize model
    print(f"Initializing model: {args.model}")
    engine_args = {k: v for k, v in vars(args).items() 
                   if k not in ['num_samples', 'few_shot_examples', 'experiment', 'output_report']}
    llm = LLM(**engine_args)
    
    # Run experiments with different few-shot configurations
    experiment_results = []
    total_start_time = time.time()
    
    for few_shot_count in EXPERIMENT_FEW_SHOT_CONFIGS:
        result = run_single_experiment(llm, texts, true_labels, few_shot_examples, few_shot_count)
        experiment_results.append(result)
        
        print(f"Completed {few_shot_count}-shot: {result['overall_accuracy']:.3f} accuracy, {result['processing_time_seconds']:.1f}s")
    
    total_time = time.time() - total_start_time
    print(f"\nTotal experiment time: {total_time:.1f} seconds")
    
    # Generate comprehensive report
    report = generate_experiment_report(experiment_results, dataset_info, args.model)
    
    # Save report
    with open(args.output_report, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {args.output_report}")
    
    # Print summary
    print_experiment_summary(report)
    
    return report


def print_experiment_summary(report):
    """Print experiment summary to console."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    summary = report['results_summary']
    best = summary['best_few_shot_config']
    stats = summary['overall_statistics']
    
    print(f"Best Configuration: {best['few_shot_count']}-shot examples")
    print(f"Best Accuracy: {best['accuracy']:.3f}")
    print(f"Mean Accuracy: {stats['mean_accuracy']:.3f} ± {stats['accuracy_std']:.3f}")
    print(f"95% Confidence Interval: [{stats['confidence_interval_95'][0]:.3f}, {stats['confidence_interval_95'][1]:.3f}]")
    
    print("\nTop 3 Performing Classes:")
    class_perf = report['analysis']['per_class_performance']
    sorted_classes = sorted(class_perf.items(), key=lambda x: x[1]['mean_f1'], reverse=True)
    
    for i, (class_id, metrics) in enumerate(sorted_classes[:3]):
        print(f"  {i+1}. {metrics['class_name']}: F1={metrics['mean_f1']:.3f}")
    
    print("\nBottom 3 Performing Classes:")
    for i, (class_id, metrics) in enumerate(sorted_classes[-3:]):
        print(f"  {i+1}. {metrics['class_name']}: F1={metrics['mean_f1']:.3f}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(report['analysis']['recommendations'], 1):
        print(f"  {i}. {rec}")


def main(args: Namespace):
    # Check if running full experiment
    if args.experiment:
        return run_full_experiment(args)
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Load patent classification data
    print("Loading patent classification dataset...")
    texts, true_labels = load_patent_dataset(args.num_samples)
    
    # Load few-shot examples
    print("Loading few-shot examples...")
    few_shot_examples = load_few_shot_examples(args.few_shot_examples)
    print(f"Loaded few-shot examples for {len(few_shot_examples)} classes")

    # Create an LLM for few-shot prompting
    print(f"Initializing model: {args.model}")
    
    # Filter out custom arguments that aren't part of EngineArgs
    engine_args = {k: v for k, v in vars(args).items() 
                   if k not in ['num_samples', 'few_shot_examples', 'experiment', 'output_report']}
    llm = LLM(**engine_args)

    # Generate few-shot prompts
    print("Creating few-shot prompts...")
    prompts = []
    for text in texts:
        prompt = create_few_shot_prompt(text, few_shot_examples)
        prompts.append(prompt)

    # Generate predictions using few-shot prompting
    print("Running few-shot classification...")
    sampling_params = SamplingParams(temperature=CLASSIFICATION_TEMPERATURE, max_tokens=MAX_CLASSIFICATION_TOKENS)
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    # Extract predictions
    predictions = []
    for output in outputs:
        response = output.outputs[0].text.strip()
        prediction = extract_prediction_from_response(response)
        predictions.append(prediction)

    # Filter out None predictions for accuracy calculation
    valid_indices = [i for i, p in enumerate(predictions) if p is not None]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_true_labels = [true_labels[i] for i in valid_indices]
    
    print(f"\nSuccessfully classified {len(valid_predictions)}/{len(predictions)} samples")
    
    # Calculate overall accuracy
    correct = sum(1 for t, p in zip(valid_true_labels, valid_predictions) if t == p)
    overall_accuracy = correct / len(valid_predictions) if valid_predictions else 0
    
    # Calculate per-class metrics
    per_class_metrics = calculate_per_class_metrics(valid_true_labels, valid_predictions)
    
    # Print results
    print(f"\nOverall Accuracy: {overall_accuracy:.3f} ({correct}/{len(valid_predictions)})")
    print("\nPer-Class Results:")
    print("=" * 100)
    print(f"{'Class':<5} {'Name':<50} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("=" * 100)
    
    for class_id in range(NUM_PATENT_CLASSES):
        metrics = per_class_metrics[class_id]
        class_name = PATENT_CLASSES[class_id][:47] + "..." if len(PATENT_CLASSES[class_id]) > 50 else PATENT_CLASSES[class_id]
        print(f"{class_id:<5} {class_name:<50} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1']:<10.3f} {metrics['support']:<10}")
    
    # Show some example predictions
    print("\nExample Classifications:")
    print("=" * 120)
    for i in range(min(5, len(valid_indices))):
        idx = valid_indices[i]
        text_preview = texts[idx][:MAX_TEXT_LENGTH_FOR_DISPLAY] + "..."
        true_class = PATENT_CLASSES[true_labels[idx]]
        pred_class = PATENT_CLASSES[valid_predictions[i]]
        status = "✓" if true_labels[idx] == valid_predictions[i] else "✗"
        
        print(f"{status} Text: {text_preview}")
        print(f"  True: {true_class}")
        print(f"  Pred: {pred_class}")
        print("-" * 120)


if __name__ == "__main__":
    args = parse_args()
    if args is not None:
        main(args)
