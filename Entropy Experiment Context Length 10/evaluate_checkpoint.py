#!/usr/bin/env python3
"""
Standalone evaluation script for loading and evaluating model checkpoints.
"""

import torch
import os
import sys
import argparse
import numpy as np
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig
from dataset_setup import DatasetManager
from model_training import ModelManager
from evaluation import Evaluator


def list_checkpoints(checkpoint_dir: str) -> List[str]:
    """List all available checkpoints in the directory."""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist!")
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt') and file.startswith('model_'):
            checkpoints.append(os.path.join(checkpoint_dir, file))
    
    return sorted(checkpoints)


def evaluate_checkpoint(checkpoint_path: str, config: ExperimentConfig, 
                       device: str = 'cuda', num_test_samples: int = 100):
    """Evaluate a single checkpoint."""
    print(f"\n{'='*60}")
    print(f"Evaluating checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"{'='*60}")
    
    # Load model from checkpoint
    model, checkpoint_info = ModelManager.load_checkpoint(checkpoint_path, device)
    
    print(f"Model configuration:")
    print(f"  Depth: {checkpoint_info['model_config']['depth']}")
    print(f"  Heads: {checkpoint_info['model_config']['heads']}")
    print(f"  Embedding dim: {checkpoint_info['model_config']['n_embd']}")
    print(f"  Training iterations: {checkpoint_info['training_info']['iterations']}")
    print(f"  Final loss: {checkpoint_info['training_info']['final_loss']:.5f}")
    print()
    
    # Setup dataset manager
    dataset_manager = DatasetManager(config)
    
    # Setup evaluator
    evaluator = Evaluator(config)
    
    # Create a dummy trainer for evaluation (we only need the model)
    from minGPT.mingpt.trainer import Trainer
    dummy_trainer = Trainer(Trainer.get_default_config(), model, None)
    
    # Setup datasets
    print("Setting up datasets...")
    mixed_loader, mixed_dataset = dataset_manager.setup_mixed_dataset()
    
    # Evaluate on fixed dataset
    print("Evaluating on fixed dataset...")
    fixed_loader = dataset_manager.create_fixed_dataset_loader()  # Uses dictionary from config
    fixed_samples = dataset_manager.collect_samples(fixed_loader)
    
    fixed_results = evaluator.evaluate_model_on_dataset(
        model, dummy_trainer, fixed_samples, 'fixed', 
        (checkpoint_info['model_config']['depth'], checkpoint_info['model_config']['heads'])
    )
    
    # Evaluate on changing dataset
    print("Evaluating on changing dataset...")
    changing_loader = dataset_manager.create_changing_dataset_loader()
    changing_samples = dataset_manager.collect_changing_samples(changing_loader, mixed_loader)
    
    changing_results = evaluator.evaluate_model_on_dataset(
        model, dummy_trainer, changing_samples, 'changing',
        (checkpoint_info['model_config']['depth'], checkpoint_info['model_config']['heads'])
    )
    
    # Print results
    print(f"\nResults for {os.path.basename(checkpoint_path)}:")
    print(f"  Fixed dataset - Mean predicted entropy: {np.mean(fixed_results['mean_predicted_entropies']):.5f}")
    print(f"  Changing dataset - Mean predicted entropy: {np.mean(changing_results['mean_predicted_entropies']):.5f}")
    print(f"  Fixed dataset - Mean true entropy: {np.mean(fixed_results['true_entropies']):.5f}")
    print(f"  Changing dataset - Mean true entropy: {np.mean(changing_results['true_entropies']):.5f}")
    
    return {
        'checkpoint_path': checkpoint_path,
        'model_config': checkpoint_info['model_config'],
        'training_info': checkpoint_info['training_info'],
        'fixed_results': fixed_results,
        'changing_results': changing_results
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate model checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='results/checkpoints',
                       help='Directory containing checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Specific checkpoint file to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    parser.add_argument('--num_test_samples', type=int, default=100,
                       help='Number of test samples for evaluation')
    parser.add_argument('--list_only', action='store_true',
                       help='Only list available checkpoints')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = ExperimentConfig()
    config.num_test_samples = args.num_test_samples
    
    if args.list_only:
        checkpoints = list_checkpoints(args.checkpoint_dir)
        if checkpoints:
            print(f"Available checkpoints in {args.checkpoint_dir}:")
            for i, checkpoint in enumerate(checkpoints, 1):
                print(f"  {i}. {os.path.basename(checkpoint)}")
        else:
            print("No checkpoints found!")
        return
    
    if args.checkpoint_path:
        # Evaluate specific checkpoint
        if not os.path.exists(args.checkpoint_path):
            print(f"Checkpoint file {args.checkpoint_path} does not exist!")
            return
        
        results = evaluate_checkpoint(args.checkpoint_path, config, args.device, args.num_test_samples)
        
        # Save results
        results_dir = os.path.join(config.results_dir, "checkpoint_evaluations")
        os.makedirs(results_dir, exist_ok=True)
        
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        results_path = os.path.join(results_dir, f"{checkpoint_name}_evaluation.json")
        
        import json
        import numpy as np
        
        def convert_numpy(obj):
            """Convert numpy arrays to lists for JSON serialization"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Convert numpy arrays to lists
        serializable_results = convert_numpy(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nEvaluation results saved to: {results_path}")
        
    else:
        # Evaluate all checkpoints
        checkpoints = list_checkpoints(args.checkpoint_dir)
        if not checkpoints:
            print("No checkpoints found!")
            return
        
        print(f"Found {len(checkpoints)} checkpoints to evaluate:")
        for checkpoint in checkpoints:
            print(f"  - {os.path.basename(checkpoint)}")
        
        all_results = []
        for checkpoint in checkpoints:
            try:
                results = evaluate_checkpoint(checkpoint, config, args.device, args.num_test_samples)
                all_results.append(results)
            except Exception as e:
                print(f"Error evaluating {checkpoint}: {e}")
                continue
        
        # Save combined results
        if all_results:
            results_dir = os.path.join(config.results_dir, "checkpoint_evaluations")
            os.makedirs(results_dir, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(results_dir, f"all_checkpoints_evaluation_{timestamp}.json")
            
            import json
            import numpy as np
            
            def convert_numpy(obj):
                """Convert numpy arrays to lists for JSON serialization"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            # Convert numpy arrays to lists
            serializable_all_results = convert_numpy(all_results)
            
            with open(results_path, 'w') as f:
                json.dump(serializable_all_results, f, indent=2)
            
            print(f"\nAll evaluation results saved to: {results_path}")
            
            # Print summary
            print(f"\n{'='*60}")
            print("EVALUATION SUMMARY")
            print(f"{'='*60}")
            for results in all_results:
                model_config = results['model_config']
                fixed = results['fixed_results']
                changing = results['changing_results']
                print(f"Depth {model_config['depth']}, Heads {model_config['heads']}: "
                      f"Fixed entropy={fixed['entropy']:.5f}, "
                      f"Changing entropy={changing['entropy']:.5f}")


if __name__ == "__main__":
    main()
