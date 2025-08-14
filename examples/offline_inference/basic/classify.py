# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
import numpy as np
from collections import Counter

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
        "--num_samples", 
        type=int, 
        default=DEFAULT_TOTAL_SAMPLES, 
        help=f"Total number of samples to classify from the dataset (default: {DEFAULT_TOTAL_SAMPLES}, ~{DEFAULT_SAMPLES_PER_CLASS} per class)"
    )
    parser.add_argument(
        "--few_shot_examples", 
        type=int, 
        default=DEFAULT_FEW_SHOT_EXAMPLES, 
        help=f"Number of examples per class for few-shot prompting (default: {DEFAULT_FEW_SHOT_EXAMPLES})"
    )
    # Set example specific arguments - using a more powerful model for few-shot classification
    parser.set_defaults(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Better model for few-shot prompting
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


def main(args: Namespace):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load patent classification data
    print("Loading patent classification dataset...")
    texts, true_labels = load_patent_dataset(args.num_samples)
    
    # Load few-shot examples
    print("Loading few-shot examples...")
    few_shot_examples = load_few_shot_examples(args.few_shot_examples)
    print(f"Loaded few-shot examples for {len(few_shot_examples)} classes")

    # Create an LLM for few-shot prompting
    print(f"Initializing model: {args.model}")
    llm = LLM(**vars(args))

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
