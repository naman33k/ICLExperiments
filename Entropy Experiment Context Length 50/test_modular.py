#!/usr/bin/env python3
"""
Test script to verify the modularized ICL experiment code works correctly.
This script performs a quick test without full training to check imports and basic functionality.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from config import ExperimentConfig
        print("✓ config.py imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import config.py: {e}")
        return False
    
    try:
        from dataset_setup import DatasetManager
        print("✓ dataset_setup.py imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import dataset_setup.py: {e}")
        return False
    
    try:
        from model_training import ModelManager
        print("✓ model_training.py imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import model_training.py: {e}")
        return False
    
    try:
        from evaluation import Evaluator
        print("✓ evaluation.py imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import evaluation.py: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration creation and basic functionality."""
    print("\nTesting configuration...")
    
    try:
        from config import ExperimentConfig
        config = ExperimentConfig()
        
        # Test basic properties
        assert config.context_length == 50
        assert config.vocab_size == 50
        assert config.depths == [6, 9, 12]
        assert config.heads == [6]
        assert config.block_size == 99  # 2*50-1
        assert config.model_vocab_size == 100  # 2*50
        
        # Test model configs
        model_configs = config.get_model_configs()
        expected_configs = [(6, 6), (9, 6), (12, 6)]
        assert model_configs == expected_configs
        
        print("✓ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_managers_initialization():
    """Test that managers can be initialized."""
    print("\nTesting manager initialization...")
    
    try:
        from config import ExperimentConfig
        from dataset_setup import DatasetManager
        from model_training import ModelManager
        from evaluation import Evaluator
        
        config = ExperimentConfig()
        
        # Test dataset manager
        dataset_manager = DatasetManager(config)
        print("✓ DatasetManager initialized")
        
        # Test model manager
        model_manager = ModelManager(config)
        print("✓ ModelManager initialized")
        
        # Test evaluator
        evaluator = Evaluator(config)
        print("✓ Evaluator initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Manager initialization test failed: {e}")
        return False

def test_results_directory():
    """Test that results directory exists and is writable."""
    print("\nTesting results directory...")
    
    try:
        from config import ExperimentConfig
        config = ExperimentConfig()
        
        # Check if results directory exists
        if os.path.exists(config.results_dir):
            print(f"✓ Results directory exists: {config.results_dir}")
        else:
            print(f"✗ Results directory does not exist: {config.results_dir}")
            return False
        
        # Check if it's writable
        test_file = os.path.join(config.results_dir, "test_write.tmp")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print("✓ Results directory is writable")
        except Exception as e:
            print(f"✗ Results directory is not writable: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Results directory test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ICL Experiment Modular Code Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_configuration,
        test_managers_initialization,
        test_results_directory
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("Test Summary")
    print("=" * 40)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The modularized code is ready to use.")
        print("\nTo run the full experiment:")
        print("  python run_icl_experiment.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
