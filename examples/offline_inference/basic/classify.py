# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

from vllm import LLM, EngineArgs
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
        default=5, 
        help="Number of samples to classify from the dataset"
    )
    # Set example specific arguments - using a more powerful model for patent classification
    parser.set_defaults(
        model="microsoft/deberta-v3-large",  # More powerful model for better classification
        runner="pooling",
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


def load_patent_dataset(num_samples=5):
    """Load patent classification dataset from HuggingFace."""
    if not DATASETS_AVAILABLE:
        print("Using fallback sample data since datasets library is not available.")
        # Fallback sample data for demonstration
        return [
            "A method for manufacturing semiconductor devices with improved efficiency...",
            "A pharmaceutical composition comprising active compounds for treating cancer...", 
            "An apparatus for wireless communication using advanced antenna arrays...",
            "A mechanical system for automotive brake control with enhanced safety...",
            "A chemical process for producing renewable energy from biomass materials..."
        ]
    
    try:
        # Load the patent classification dataset
        dataset = load_dataset("ccdv/patent-classification", split="test")
        
        # Sample a few examples for classification
        samples = dataset.shuffle(seed=42).select(range(num_samples))
        
        # Extract text and labels
        texts = samples["text"]
        labels = samples["label"] 
        
        print(f"Loaded {len(texts)} samples from patent classification dataset")
        print("True labels:", [PATENT_CLASSES[label] for label in labels])
        
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
        ], None


def main(args: Namespace):
    # Load patent classification data
    print("Loading patent classification dataset...")
    dataset_result = load_patent_dataset(args.num_samples)
    
    if isinstance(dataset_result, tuple):
        # We have texts and true labels
        texts, true_labels = dataset_result
    else:
        # Only texts (fallback case)
        texts = dataset_result
        true_labels = None
    
    # Truncate texts if they're too long (keep first 1000 characters for demonstration)
    processed_texts = []
    for text in texts:
        if len(text) > 1000:
            processed_text = text[:1000] + "..."
            print(f"Truncated text from {len(text)} to 1000 characters")
        else:
            processed_text = text
        processed_texts.append(processed_text)

    # Create an LLM for classification
    # Using runner="pooling" for classification models
    print(f"Initializing model: {args.model}")
    llm = LLM(**vars(args))

    # Generate classifications. The output is a list of ClassificationRequestOutputs.
    print("Running classification...")
    outputs = llm.classify(processed_texts)

    # Print the results with patent class interpretations
    print("\nPatent Classification Results:\n" + "=" * 80)
    for i, (text, output) in enumerate(zip(processed_texts, outputs)):
        probs = output.outputs.probs
        
        # Find top 3 predicted classes
        top_indices = sorted(range(len(probs)), key=lambda x: probs[x], reverse=True)[:3]
        
        print(f"Sample {i+1}:")
        print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"True Label: {PATENT_CLASSES[true_labels[i]] if true_labels else 'N/A'}")
        print("Top Predictions:")
        
        for j, idx in enumerate(top_indices):
            class_name = PATENT_CLASSES[idx]
            confidence = probs[idx] * 100
            print(f"  {j+1}. {class_name}: {confidence:.2f}%")
        
        print("-" * 80)


if __name__ == "__main__":
    args = parse_args()
    if args is not None:
        main(args)
