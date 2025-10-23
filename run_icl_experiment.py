#!/usr/bin/env python3
"""
Main script for running ICL experiments.
Orchestrates dataset setup, model training, and evaluation.
"""

import torch
import sys
import os
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig
from dataset_setup import DatasetManager
from model_training import ModelManager
from evaluation import Evaluator


def main():
    """Main function to run the complete ICL experiment."""
    
    print("Starting ICL Experiment")
    print("=" * 50)
    
    # Initialize configuration
    config = ExperimentConfig()
    print(f"Configuration loaded:")
    print(f"  Context length: {config.context_length}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Model depths: {config.depths}")
    print(f"  Model heads: {config.heads}")
    print(f"  Results directory: {config.results_dir}")
    print()
    
    # Check CUDA availability
    print(f"CUDA devices available: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")
    print()
    
    # Initialize managers
    dataset_manager = DatasetManager(config)
    model_manager = ModelManager(config)
    evaluator = Evaluator(config)
    
    # Setup dataset
    print("Setting up datasets...")
    mixed_loader, mixed_dataset = dataset_manager.setup_mixed_dataset()
    
    # Calculate theoretical entropies
    ideal_entropy, dumb_entropy = dataset_manager.calculate_theoretical_entropies()
    print(f"Theoretical entropies:")
    print(f"  Ideal entropy: {ideal_entropy:.5f}")
    print(f"  Dumb entropy: {dumb_entropy:.5f}")
    print()
    
    # Train models
    print("Training models...")
    model_dicts, trainer_dicts = model_manager.train_models(mixed_dataset)
    print("Model training completed!")
    print()
    
    # Evaluation phase
    print("Starting evaluation...")
    all_results = []
    
    # Get all model configurations
    model_configs = config.get_model_configs()
    
    for depth, heads in model_configs:
        print(f"\nEvaluating model: depth={depth}, heads={heads}")
        
        # Get trained model and trainer
        model, trainer = model_manager.get_model(depth, heads)
        
        # Evaluate on fixed dataset
        print("  Evaluating on fixed dataset...")
        fixed_loader = dataset_manager.create_fixed_dataset_loader(mixed_dataset.dictionary)
        fixed_samples = dataset_manager.collect_samples(fixed_loader)
        
        fixed_results = evaluator.evaluate_model_on_dataset(
            model, trainer, fixed_samples, 'fixed', (depth, heads)
        )
        all_results.append(fixed_results)
        
        # Evaluate on changing dataset
        print("  Evaluating on changing dataset...")
        changing_loader = dataset_manager.create_changing_dataset_loader()
        changing_samples = dataset_manager.collect_changing_samples(
            changing_loader, mixed_loader
        )
        
        changing_results = evaluator.evaluate_model_on_dataset(
            model, trainer, changing_samples, 'changing', (depth, heads)
        )
        all_results.append(changing_results)
    
    # Save results summary
    print("\nSaving results summary...")
    evaluator.save_results_summary(all_results)
    
    print("\nExperiment completed successfully!")
    print(f"All results saved in: {config.results_dir}")


if __name__ == "__main__":
    main()
