#!/usr/bin/env python3
"""
DDP-specific runner for ICL experiments.
This script is designed to be launched with torch.distributed.launch.
"""

import torch
import torch.distributed as dist
import os
import sys
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig
from dataset_setup import DatasetManager
from model_training import ModelManager
from evaluation import Evaluator


def main():
    """Main function for DDP training."""
    
    # Initialize DDP
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("DDP environment variables not found. Exiting.")
        return
    
    # Initialize the process group
    dist.init_process_group(backend='nccl')
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        print(f"Starting DDP training with {world_size} processes")
        print(f"Local rank: {local_rank}, Global rank: {rank}")
    
    # Initialize configuration
    config = ExperimentConfig()
    
    if rank == 0:
        print(f"Configuration loaded:")
        print(f"  Context length: {config.context_length}")
        print(f"  Vocab size: {config.vocab_size}")
        print(f"  Model depths: {config.depths}")
        print(f"  Model heads: {config.heads}")
        print(f"  Results directory: {config.results_dir}")
        print()
    
    # Initialize managers
    dataset_manager = DatasetManager(config)
    model_manager = ModelManager(config, use_ddp=True)
    evaluator = Evaluator(config)
    
    # Setup dataset
    if rank == 0:
        print("Setting up datasets...")
    mixed_loader, mixed_dataset = dataset_manager.setup_mixed_dataset()
    
    # Calculate theoretical entropies (only on rank 0)
    if rank == 0:
        ideal_entropy, dumb_entropy = dataset_manager.calculate_theoretical_entropies()
        print(f"Theoretical entropies:")
        print(f"  Ideal entropy: {ideal_entropy:.5f}")
        print(f"  Dumb entropy: {dumb_entropy:.5f}")
        print()
    
    # Train models
    if rank == 0:
        print("Training models...")
    model_dicts, trainer_dicts = model_manager.train_models(mixed_dataset)
    
    if rank == 0:
        print("Model training completed!")
        print()
    
    # Evaluation phase (only on rank 0 to avoid conflicts)
    if rank == 0:
        print("Starting evaluation...")
        all_results = []
        
        # Get all model configurations
        model_configs = config.get_model_configs()
        
        for depth, heads in model_configs:
            print(f"\nEvaluating model: depth={depth}, heads={heads}")
            
            # Get trained model and trainer
            model, trainer = model_manager.get_model(depth, heads)
            
            # Unwrap DDP model for evaluation
            model = model_manager.unwrap_ddp_model(model)
            
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
        
        print("\nDDP experiment completed successfully!")
        print(f"All results saved in: {config.results_dir}")
    
    # Clean up DDP
    model_manager.cleanup_ddp()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
