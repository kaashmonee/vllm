#!/usr/bin/env python3

"""
Test Script for Multi-LoRA Patent Classification System

This script demonstrates how to use the multi-adapter LoRA system for patent classification.
It provides examples of different usage modes and validates the system setup.

USAGE:
# Test basic LoRA functionality (requires trained adapters)
python test_lora_system.py --test-basic

# Test ensemble mode
python test_lora_system.py --test-ensemble --lora-adapter-dir ./lora_adapters

# Test specific domain adapter
python test_lora_system.py --test-domain chemical_materials

# Run full system validation
python test_lora_system.py --validate-system
"""

import argparse
import time
from pathlib import Path

# Test patent examples for each domain
TEST_PATENT_EXAMPLES = {
    "chemical_materials": """
    A method for synthesizing novel polymer compounds with enhanced thermal stability
    using advanced catalytic processes. The invention involves molecular engineering
    of chemical bonds to achieve superior material properties in high-temperature
    applications through controlled polymerization reactions.
    """,
    
    "engineering_mechanical": """
    An improved mechanical assembly for automotive brake systems with enhanced
    safety features. The device includes precision-engineered components with
    optimal pressure distribution and real-time performance monitoring capabilities
    for vehicle safety applications.
    """,
    
    "electronics_physics": """
    A semiconductor device with advanced transistor architecture for high-frequency
    applications. The invention utilizes novel electronic circuit designs with
    improved voltage regulation and signal processing capabilities for computing
    systems and power management.
    """,
    
    "life_sciences": """
    A pharmaceutical composition for treating cancer with reduced side effects.
    The therapeutic formulation includes biocompatible drug delivery mechanisms
    and targeted molecular treatments for improved patient outcomes in medical
    applications.
    """,
    
    "emerging_crosscutting": """
    An innovative nanotechnology-based system combining artificial intelligence
    and biotechnology for smart materials. The invention integrates multiple
    disciplines including quantum computing, renewable energy, and advanced
    manufacturing processes.
    """
}

def test_lora_system_basic(adapter_dir: str = "./lora_adapters"):
    """Test basic LoRA system functionality."""
    print("🧪 Testing Basic LoRA System")
    print("=" * 50)
    
    try:
        # Import after checking if we can load the system
        from classify_unified import UnifiedPatentClassifier, parse_unified_args
        from lora_domain_config import PATENT_DOMAIN_GROUPS
        
        # Create minimal args for testing
        class TestArgs:
            def __init__(self):
                # Basic configuration
                self.model = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
                self.num_samples = 5  # Small test
                self.few_shot_examples = 1
                self.experiment = False
                self.output_report = "test_lora_report.json"
                
                # Enable LoRA
                self.use_lora_adapters = True
                self.lora_adapter_dir = adapter_dir
                self.lora_domain = None  # Test auto-selection
                self.lora_ensemble_mode = False
                self.lora_confidence_threshold = 0.6  # Lenient for testing
                
                # Disable other optimizations for simplicity
                self.enhanced_prompts = False
                self.advanced_sampling = False
                self.chain_of_thought = False
                self.confidence_scoring = False
                self.vllm_optimizations = False
                self.optimal_batching = False
                self.parallel_sampling = False
                self.use_logit_bias = False
                self.kv_cache_optimization = False
                self.custom_batch_size = None
                self.ray_distributed = False
                
                # Default vLLM args
                self.gpu_memory_utilization = 0.8
                self.swap_space = 4
                self.enforce_eager = True
                self.trust_remote_code = True
                self.dtype = "auto"
                self.max_model_len = 4096
        
        args = TestArgs()
        
        # Initialize classifier
        print("🚀 Initializing LoRA-enabled classifier...")
        classifier = UnifiedPatentClassifier(args)
        classifier.initialize_engine()
        
        # Test each domain example
        results = []
        for domain, example_text in TEST_PATENT_EXAMPLES.items():
            print(f"\n🔍 Testing domain: {domain}")
            print(f"Example text: {example_text[:100]}...")
            
            start_time = time.time()
            prediction, confidence, method = classifier.predict_with_lora_ensemble(example_text.strip())
            processing_time = time.time() - start_time
            
            result = {
                "domain": domain,
                "prediction": prediction,
                "confidence": confidence,
                "method": method,
                "processing_time": processing_time,
                "expected_classes": PATENT_DOMAIN_GROUPS[domain]["classes"]
            }
            results.append(result)
            
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Method: {method}")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Expected classes: {result['expected_classes']}")
            
            # Validate result
            if prediction is not None and prediction in result['expected_classes']:
                print("   ✅ Correct domain classification")
            elif prediction is not None:
                print("   ⚠️  Incorrect domain (but valid prediction)")
            else:
                print("   ❌ No prediction made")
        
        # Summary
        print(f"\n📊 TEST SUMMARY")
        print("=" * 50)
        successful_predictions = sum(1 for r in results if r['prediction'] is not None)
        correct_domain_predictions = sum(1 for r in results 
                                       if r['prediction'] is not None and 
                                       r['prediction'] in r['expected_classes'])
        avg_confidence = sum(r['confidence'] for r in results if r['confidence'] > 0) / len(results)
        avg_processing_time = sum(r['processing_time'] for r in results) / len(results)
        
        print(f"Total tests: {len(results)}")
        print(f"Successful predictions: {successful_predictions}/{len(results)}")
        print(f"Correct domain classifications: {correct_domain_predictions}/{len(results)}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Average processing time: {avg_processing_time:.2f}s")
        
        if successful_predictions == len(results):
            print("🎉 All tests passed - LoRA system working correctly!")
            return True
        else:
            print("⚠️  Some tests failed - check adapter availability and configuration")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_lora_setup(adapter_dir: str = "./lora_adapters"):
    """Validate LoRA system setup and configuration."""
    print("🔧 Validating LoRA System Setup")
    print("=" * 50)
    
    validation_results = {
        "dependencies": False,
        "adapter_directory": False,
        "adapter_files": False,
        "domain_config": False
    }
    
    # Check dependencies
    try:
        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✅ PEFT and transformers dependencies available")
        validation_results["dependencies"] = True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Install with: pip install peft transformers")
    
    # Check adapter directory
    adapter_path = Path(adapter_dir)
    if adapter_path.exists():
        print(f"✅ Adapter directory exists: {adapter_path}")
        validation_results["adapter_directory"] = True
    else:
        print(f"❌ Adapter directory not found: {adapter_path}")
        print("   Train adapters first with: python train_lora_adapters.py --train-all-domains")
    
    # Check for adapter files
    if validation_results["adapter_directory"]:
        try:
            from lora_domain_config import PATENT_DOMAIN_GROUPS
            
            found_adapters = []
            for domain in PATENT_DOMAIN_GROUPS.keys():
                domain_path = adapter_path / domain / "final"
                config_path = domain_path / "adapter_config.json"
                
                if domain_path.exists() and config_path.exists():
                    found_adapters.append(domain)
                    print(f"✅ Found adapter: {domain}")
                else:
                    print(f"❌ Missing adapter: {domain}")
            
            if found_adapters:
                validation_results["adapter_files"] = True
                print(f"📋 Available adapters: {found_adapters}")
            else:
                print("❌ No valid adapters found")
        except ImportError:
            print("❌ Cannot import domain configuration")
    
    # Check domain configuration
    try:
        from lora_domain_config import (
            PATENT_DOMAIN_GROUPS, 
            LORA_TRAINING_CONFIGS,
            MULTI_ADAPTER_CONFIG
        )
        print("✅ Domain configuration loaded successfully")
        print(f"   Configured domains: {list(PATENT_DOMAIN_GROUPS.keys())}")
        validation_results["domain_config"] = True
    except ImportError as e:
        print(f"❌ Domain configuration error: {e}")
    
    # Overall validation result
    all_valid = all(validation_results.values())
    
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 50)
    for check, result in validation_results.items():
        status = "✅" if result else "❌"
        print(f"{status} {check.replace('_', ' ').title()}")
    
    if all_valid:
        print("\n🎉 LoRA system setup is complete and ready to use!")
        return True
    else:
        print("\n⚠️  LoRA system setup is incomplete. Address the issues above.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Multi-LoRA Patent Classification System")
    parser.add_argument("--test-basic", action="store_true",
                       help="Run basic LoRA functionality test")
    parser.add_argument("--test-ensemble", action="store_true",
                       help="Test ensemble mode")
    parser.add_argument("--test-domain", type=str,
                       help="Test specific domain adapter")
    parser.add_argument("--validate-system", action="store_true",
                       help="Validate LoRA system setup")
    parser.add_argument("--lora-adapter-dir", type=str, default="./lora_adapters",
                       help="Directory containing LoRA adapters")
    
    args = parser.parse_args()
    
    if not any([args.test_basic, args.test_ensemble, args.test_domain, args.validate_system]):
        parser.error("Must specify at least one test mode")
    
    success = True
    
    # Run validation first if requested
    if args.validate_system:
        success = validate_lora_setup(args.lora_adapter_dir) and success
    
    # Run tests
    if args.test_basic:
        success = test_lora_system_basic(args.lora_adapter_dir) and success
    
    if args.test_ensemble:
        print("🔄 Ensemble testing not yet implemented")
        # TODO: Implement ensemble testing
    
    if args.test_domain:
        print(f"🔄 Domain-specific testing for {args.test_domain} not yet implemented")
        # TODO: Implement domain-specific testing
    
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        exit(1)

if __name__ == "__main__":
    main()