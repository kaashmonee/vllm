#!/usr/bin/env python3

"""
LoRA Domain Configuration for Patent Classification

This module defines domain groupings for training specialized LoRA adapters
based on patent classification performance analysis and technical similarity.

Domain groupings are designed to:
1. Group technically similar patent classes for better specialization
2. Balance training data across domains 
3. Optimize for classes that showed poor performance in baseline experiments
"""

# Domain groupings based on technical similarity and performance analysis
PATENT_DOMAIN_GROUPS = {
    # High-performing chemical/materials domain
    "chemical_materials": {
        "classes": [2, 3],  # Chemistry/Metallurgy, Textiles/Paper
        "description": "Chemical processes, materials, compounds, textiles, fibers, paper",
        "keywords": [
            "chemical", "compound", "polymer", "catalyst", "synthesis", "molecular", 
            "alloy", "metallurgy", "textile", "fabric", "fiber", "yarn", "weaving", 
            "paper", "pulp", "cloth", "thread", "material", "coating", "adhesive"
        ],
        "target_classes": {2: "Chemistry; Metallurgy", 3: "Textiles; Paper"},
        "baseline_performance": {"chemistry": 0.586, "textiles": 0.503}  # F1 scores
    },
    
    # Engineering/mechanical systems domain  
    "engineering_mechanical": {
        "classes": [1, 4, 5],  # Operations/Transport, Fixed Construction, Mechanical Engineering
        "description": "Manufacturing, transportation, construction, mechanical systems",
        "keywords": [
            "machine", "manufacturing", "vehicle", "transport", "engine", "pump", 
            "assembly", "production", "building", "construction", "structure", 
            "foundation", "concrete", "beam", "roof", "bridge", "road", "mechanical",
            "gear", "bearing", "valve", "heating", "cooling", "turbine", "hvac"
        ],
        "target_classes": {
            1: "Performing Operations; Transporting",
            4: "Fixed Constructions", 
            5: "Mechanical Engineering; Lightning; Heating; Weapons; Blasting"
        },
        "baseline_performance": {"operations": 0.356, "construction": 0.488, "mechanical": 0.417}
    },
    
    # Electronics/physics domain
    "electronics_physics": {
        "classes": [6, 7],  # Physics, Electricity
        "description": "Electronic devices, circuits, optics, measurements, power systems",
        "keywords": [
            "optical", "lens", "camera", "measurement", "sensor", "detector", "laser",
            "instrument", "electrical", "electronic", "circuit", "power", "voltage",
            "battery", "semiconductor", "computer", "processor", "memory", "display"
        ],
        "target_classes": {6: "Physics", 7: "Electricity"},
        "baseline_performance": {"physics": 0.371, "electricity": 0.533}
    },
    
    # Life sciences/human domain
    "life_sciences": {
        "classes": [0],  # Human Necessities
        "description": "Medical, pharmaceutical, food, consumer products, health",
        "keywords": [
            "food", "beverage", "medical", "health", "drug", "pharmaceutical",
            "clothing", "game", "sport", "household", "therapeutic", "diagnostic",
            "biological", "biotechnology", "medical device", "treatment"
        ],
        "target_classes": {0: "Human Necessities"},
        "baseline_performance": {"human_necessities": 0.573}
    },
    
    # Cross-cutting/emerging technologies (most challenging)
    "emerging_crosscutting": {
        "classes": [8],  # General tagging
        "description": "Emerging technologies, nanotechnology, cross-disciplinary innovations",
        "keywords": [
            "nanotechnology", "biotechnology", "emerging", "novel", "innovative", 
            "advanced", "interdisciplinary", "cross-sectional", "hybrid", "smart",
            "artificial intelligence", "machine learning", "quantum", "renewable"
        ],
        "target_classes": {8: "General tagging of new or cross-sectional technology"},
        "baseline_performance": {"general_crosscutting": 0.077}  # Needs most help
    }
}

# LoRA configuration constants with rationale
# Rank values chosen based on domain complexity and expected parameter efficiency
LORA_HIGH_COMPLEXITY_RANK = 32    # For domains with rich technical vocabulary (chemistry)
LORA_MEDIUM_COMPLEXITY_RANK = 24   # For moderate technical complexity (engineering)
LORA_STANDARD_COMPLEXITY_RANK = 20  # For standard domains (life sciences)
LORA_LOW_COMPLEXITY_RANK = 16      # For problematic/sparse data domains (cross-cutting)

# Alpha values - typically 2x rank for balanced adaptation strength
LORA_HIGH_ALPHA = 64    # 2x high rank
LORA_MEDIUM_ALPHA = 48  # 2x medium rank
LORA_STANDARD_ALPHA = 40 # 2x standard rank
LORA_LOW_ALPHA = 32     # 2x low rank

# Dropout rates based on data quality and overfitting risk
LORA_LOW_DROPOUT = 0.05     # For stable, high-quality data (chemistry)
LORA_STANDARD_DROPOUT = 0.1  # Standard dropout rate for most domains
LORA_HIGH_DROPOUT = 0.15    # For noisy/sparse data to prevent overfitting

# Learning rates optimized for LoRA fine-tuning stability
LORA_CONSERVATIVE_LR = 1e-4  # Conservative rate for challenging domains
LORA_STANDARD_LR = 2e-4      # Standard rate for most domains  
LORA_AGGRESSIVE_LR = 3e-4    # Slightly higher rate for well-behaved domains

# Training samples per class - balanced for data availability and training time
CHEMISTRY_TRAINING_SAMPLES = 300    # Rich dataset, complex terminology
ENGINEERING_TRAINING_SAMPLES = 250  # Moderate dataset size
ELECTRONICS_TRAINING_SAMPLES = 350  # Large dataset due to Physics class size
LIFE_SCIENCES_TRAINING_SAMPLES = 400 # Single class, needs more intensive training
CROSSCUTTING_TRAINING_SAMPLES = 500 # Poor performance, needs maximum data

# Training epochs based on domain stability and convergence patterns
STABLE_DOMAIN_EPOCHS = 4      # For well-behaved domains
STANDARD_DOMAIN_EPOCHS = 5    # Standard epoch count
INTENSIVE_TRAINING_EPOCHS = 6  # For challenging single-class domains
MAXIMUM_TRAINING_EPOCHS = 8   # For most challenging domains

# LoRA training configuration for each domain
LORA_TRAINING_CONFIGS = {
    "chemical_materials": {
        "rank": LORA_HIGH_COMPLEXITY_RANK,  # Complex chemical terminology needs high rank
        "alpha": LORA_HIGH_ALPHA,
        "dropout": LORA_LOW_DROPOUT,  # Chemistry data is stable and high-quality
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "training_samples_per_class": CHEMISTRY_TRAINING_SAMPLES,
        "learning_rate": LORA_STANDARD_LR,
        "epochs": STANDARD_DOMAIN_EPOCHS
    },
    
    "engineering_mechanical": {
        "rank": LORA_MEDIUM_COMPLEXITY_RANK,  # Moderate complexity for mechanical concepts
        "alpha": LORA_MEDIUM_ALPHA,
        "dropout": LORA_STANDARD_DROPOUT,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Standard attention modules
        "training_samples_per_class": ENGINEERING_TRAINING_SAMPLES,
        "learning_rate": LORA_AGGRESSIVE_LR,  # Higher rate for well-behaved domains
        "epochs": STABLE_DOMAIN_EPOCHS
    },
    
    "electronics_physics": {
        "rank": LORA_MEDIUM_COMPLEXITY_RANK,  # Physics/electronics have good technical vocabulary
        "alpha": LORA_MEDIUM_ALPHA + 8,  # Slightly higher alpha for better adaptation
        "dropout": LORA_STANDARD_DROPOUT,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],  # Include gate projection
        "training_samples_per_class": ELECTRONICS_TRAINING_SAMPLES,  # Larger due to Physics class size
        "learning_rate": LORA_STANDARD_LR,
        "epochs": INTENSIVE_TRAINING_EPOCHS
    },
    
    "life_sciences": {
        "rank": LORA_STANDARD_COMPLEXITY_RANK,  # Single class domain needs focused approach
        "alpha": LORA_STANDARD_ALPHA,
        "dropout": LORA_STANDARD_DROPOUT,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "training_samples_per_class": LIFE_SCIENCES_TRAINING_SAMPLES,  # Intensive single-class training
        "learning_rate": LORA_STANDARD_LR,
        "epochs": INTENSIVE_TRAINING_EPOCHS
    },
    
    "emerging_crosscutting": {
        "rank": LORA_LOW_COMPLEXITY_RANK,  # Lower rank due to data quality issues
        "alpha": LORA_LOW_ALPHA,
        "dropout": LORA_HIGH_DROPOUT,  # Prevent overfitting on poor-quality data
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "training_samples_per_class": CROSSCUTTING_TRAINING_SAMPLES,  # Maximum data for challenging class
        "learning_rate": LORA_CONSERVATIVE_LR,  # Conservative rate for stability
        "epochs": MAXIMUM_TRAINING_EPOCHS
    }
}

# Multi-adapter inference constants with rationale
CONFIDENCE_THRESHOLD_DEFAULT = 0.7     # 70% confidence threshold - balances accuracy vs coverage
CONFIDENCE_THRESHOLD_STRICT = 0.8      # Higher threshold for critical applications
CONFIDENCE_THRESHOLD_LENIENT = 0.6     # Lower threshold for broader coverage

MAX_ADAPTERS_TO_TRY = 3                # Optimal balance of coverage vs computational cost
ENSEMBLE_GENERATION_SAMPLES = 5        # Multiple samples per adapter for confidence estimation
ADAPTER_KEYWORD_MATCH_THRESHOLD = 1    # Minimum keyword matches to consider adapter

# Temperature settings for adapter inference
ADAPTER_TEMPERATURE_DETERMINISTIC = 0.0  # For single best prediction
ADAPTER_TEMPERATURE_SAMPLING = 0.3       # For confidence estimation via multiple samples

# Text processing limits for adapter inference
ADAPTER_MAX_INPUT_LENGTH = 600           # Truncate long patents for consistent processing
ADAPTER_RESPONSE_MAX_LENGTH = 20         # Limit response parsing to first 20 characters

# Ensemble voting configuration
ENSEMBLE_CONFIDENCE_WEIGHT_EXPONENT = 2  # Square confidence to emphasize high-confidence predictions
ENSEMBLE_MIN_ADAPTERS = 2                # Minimum adapters needed for valid ensemble
ENSEMBLE_MAX_ADAPTERS = 4                # Maximum adapters to prevent excessive computation

# Inference configuration for multi-adapter system
MULTI_ADAPTER_CONFIG = {
    "confidence_threshold": CONFIDENCE_THRESHOLD_DEFAULT,  # Balanced confidence threshold
    "max_adapters_to_try": MAX_ADAPTERS_TO_TRY,  # Optimal coverage vs speed tradeoff
    "adapter_selection_method": "confidence_weighted",  # Best performing selection method
    "fallback_to_base_model": True,  # Safety fallback for edge cases
    "ensemble_voting": False,  # Default to single-best for speed
    "generation_samples": ENSEMBLE_GENERATION_SAMPLES,  # Samples for confidence estimation
    "generation_temperature": ADAPTER_TEMPERATURE_SAMPLING,  # Temperature for sampling
    "max_input_length": ADAPTER_MAX_INPUT_LENGTH,  # Input truncation limit
    "response_parse_length": ADAPTER_RESPONSE_MAX_LENGTH,  # Response parsing limit
}

# Domain routing keywords for fast adapter selection
DOMAIN_ROUTING_KEYWORDS = {}
for domain, config in PATENT_DOMAIN_GROUPS.items():
    DOMAIN_ROUTING_KEYWORDS[domain] = {
        "primary": config["keywords"][:10],  # Top 10 most distinctive keywords
        "secondary": config["keywords"][10:],  # Additional keywords
        "weight": len(config["classes"])  # Weight by number of classes in domain
    }

def get_domain_for_class(patent_class: int) -> str:
    """Get the domain name for a given patent class."""
    for domain, config in PATENT_DOMAIN_GROUPS.items():
        if patent_class in config["classes"]:
            return domain
    return "unknown"

def get_classes_for_domain(domain: str) -> list:
    """Get all patent classes for a given domain."""
    if domain in PATENT_DOMAIN_GROUPS:
        return PATENT_DOMAIN_GROUPS[domain]["classes"]
    return []

def get_domain_keywords(domain: str) -> list:
    """Get keywords for a given domain."""
    if domain in PATENT_DOMAIN_GROUPS:
        return PATENT_DOMAIN_GROUPS[domain]["keywords"]
    return []

def route_patent_to_domain(patent_text: str) -> str:
    """Simple keyword-based routing of patent to most relevant domain."""
    text_lower = patent_text.lower()
    domain_scores = {}
    
    for domain, keywords in DOMAIN_ROUTING_KEYWORDS.items():
        score = 0
        # Primary keywords get higher weight
        for keyword in keywords["primary"]:
            if keyword in text_lower:
                score += 3
        # Secondary keywords get lower weight  
        for keyword in keywords["secondary"]:
            if keyword in text_lower:
                score += 1
        
        # Weight by domain size
        score *= keywords["weight"]
        domain_scores[domain] = score
    
    # Return domain with highest score, or default to engineering_mechanical
    if domain_scores:
        return max(domain_scores, key=domain_scores.get)
    return "engineering_mechanical"  # Default for unknown patents

if __name__ == "__main__":
    # Print domain configuration summary
    print("Patent Classification LoRA Domain Configuration")
    print("=" * 60)
    
    for domain, config in PATENT_DOMAIN_GROUPS.items():
        print(f"\nDomain: {domain}")
        print(f"  Classes: {config['classes']}")
        print(f"  Description: {config['description']}")
        print(f"  Target Classes: {list(config['target_classes'].values())}")
        print(f"  Baseline F1 Scores: {config['baseline_performance']}")
        print(f"  LoRA Rank: {LORA_TRAINING_CONFIGS[domain]['rank']}")
        print(f"  Training Samples/Class: {LORA_TRAINING_CONFIGS[domain]['training_samples_per_class']}")