"""
Evaluation module for ICL experiments.
Handles entropy calculations, plotting, and result saving.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import entropy
from torch.nn import functional as F
from typing import List, Tuple, Dict, Any
import os
from datetime import datetime
from config import ExperimentConfig


class Evaluator:
    """Handles model evaluation, entropy calculations, and result visualization."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_plotting()
        
    def setup_plotting(self):
        """Set up plotting style and parameters."""
        plt.style.use('default')
        sns.set_palette("husl")
        
    def calculate_predicted_entropies(self, model, trainer, samples: List) -> np.ndarray:
        """Calculate entropy of predicted distributions for given samples."""
        all_ents = []
        
        for a, b in samples:
            # print(a, b)
            # # Ensure input sequence fits within block size
            # max_input_len = model.block_size - 1  # Leave room for generated token
            # print(a.size(1), max_input_len, model.block_size)
            # if a.size(1) > max_input_len:
            #     a = a[:, -max_input_len:]  # Truncate from the left
            
            # Get logits for the input sequence
            with torch.no_grad():
                logits, _ = model(a.to(trainer.device))
                # Get logits for all positions
                probs = F.softmax(logits, dim=-1)  # Shape: [batch_size, seq_len, vocab_size]
            
            # Calculate entropy only for value positions (even indices: 0, 2, 4, ...)
            # This matches the structure: [k1, v1, k2, v2, ..., kn, vn]
            sample_entropies = []
            for i in range(probs.size(0)):  # For each sample in batch
                position_entropies = []
                for j in range(0, probs.size(1), 2):  # Only value positions (even indices)
                    position_entropies.append(entropy(probs[i, j].cpu().numpy()))
                sample_entropies.append(position_entropies)
            all_ents.append(sample_entropies)
            # print(a, b, sample_entropies)
            break
            
        return np.array(all_ents)
    
    def calculate_true_entropies_fixed(self, samples: List) -> List[float]:
        """Calculate true conditional distribution entropies for fixed dataset."""
        l = self.config.context_length
        true_entropies = []
        
        # Use the last sample to get the structure
        a, b = samples[-1]
        
        for i in range(0, a[0].size(0), 2):
            q = a[0][i].cpu().item()
            qa = b[0][i].cpu().item()
            j = i // 2
            
            pa = (0.9 + 0.1 * math.factorial(l - (j + 1)) / math.factorial(l)) / (0.9 + 0.1 * math.factorial(l - j) / math.factorial(l))
            pb = (1 - pa) / (l - 1)
            
            true_entropies.append(entropy([pa] + [pb] * (l - 1)))
            
        return true_entropies
    
    def calculate_true_entropies_changing(self, samples: List) -> List[float]:
        """Calculate true conditional distribution entropies for changing dataset."""
        l = self.config.context_length
        true_entropies = []
        
        # Use the last sample to get the structure
        a, b = samples[-1]
        
        for i in range(0, a[0].size(0), 2):
            if i == 0:
                q = a[0][i].cpu().item()
                qa = b[0][i].cpu().item()
                j = i // 2
                pa = (0.9 + 0.1 * math.factorial(l - (j + 1)) / math.factorial(l)) / (0.9 + 0.1 * math.factorial(l - j) / math.factorial(l))
                pb = (1 - pa) / (l - 1)
                true_entropies.append(entropy([pa] + [pb] * (l - 1)))
            else:
                true_entropies.append(np.log(l - i // 2))
                
        return true_entropies
    
    def plot_entropy_comparison(self, predicted_entropies: np.ndarray, 
                              true_entropies: List[float], 
                              model_config: Tuple[int, int],
                              dataset_type: str,
                              save_plot: bool = True) -> None:
        """Plot comparison between predicted and true entropies."""
        depth, heads = model_config
        l = self.config.context_length
        
        # Ensure predicted_entropies is 2D: [num_samples, num_positions]
        if predicted_entropies.ndim == 3:
            # If it's 3D, we need to reshape it
            predicted_entropies = predicted_entropies.reshape(predicted_entropies.shape[0], -1)
        
        # Calculate statistics
        stds = np.std(predicted_entropies, axis=0)
        mean_entropies = np.mean(predicted_entropies, axis=0)
        
        # Ensure arrays are 1D
        if stds.ndim > 1:
            stds = stds.flatten()
        if mean_entropies.ndim > 1:
            mean_entropies = mean_entropies.flatten()
        
        # Ensure dimensions match
        min_len = min(len(mean_entropies), len(true_entropies))
        mean_entropies = mean_entropies[:min_len]
        stds = stds[:min_len]
        true_entropies = true_entropies[:min_len]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot predicted entropies with error bars
        plt.fill_between(range(len(mean_entropies)), 
                        mean_entropies - stds, 
                        mean_entropies + stds, 
                        alpha=0.3, 
                        label='±1 Std Dev Predicted Entropy')
        
        plt.plot(mean_entropies, 
                label=f'Average Entropy of Predicted Distribution (depth={depth}, heads={heads})', 
                linewidth=2)
        
        plt.plot(true_entropies, 
                label=f'True Conditional Distribution', 
                linewidth=2, 
                linestyle='--')
        
        plt.xlabel('Label Position w_i')
        plt.ylabel('Entropy')
        plt.title(f'Entropy Comparison - {dataset_type.title()} Dataset (l={l})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, len(mean_entropies), 5))
        
        # Set y-axis limits to accommodate both predicted and true entropies
        max_pred = np.max(mean_entropies + stds)
        max_true = np.max(true_entropies)
        min_pred = np.max([0, np.min(mean_entropies - stds)])
        min_true = np.min(true_entropies)
        
        y_max = max(max_pred, max_true) + 0.5
        y_min = min(min_pred, min_true) - 0.1
        
        plt.ylim(y_min, y_max)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"entropy_comparison_{dataset_type}_depth{depth}_heads{heads}_{timestamp}.png"
            filepath = os.path.join(self.config.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
        
        plt.show()
    
    def evaluate_model_on_dataset(self, model, trainer, samples: List, 
                                dataset_type: str, model_config: Tuple[int, int]) -> Dict[str, Any]:
        """Evaluate a single model on a dataset and return results."""
        # Calculate entropies
        predicted_entropies = self.calculate_predicted_entropies(model, trainer, samples)
        
        if dataset_type == 'fixed':
            true_entropies = self.calculate_true_entropies_fixed(samples)
        else:  # changing
            true_entropies = self.calculate_true_entropies_changing(samples)
        
        # Plot results
        self.plot_entropy_comparison(predicted_entropies, true_entropies, 
                                   model_config, dataset_type, 
                                   save_plot=self.config.save_plots)
        
        # Return results
        return {
            'predicted_entropies': predicted_entropies,
            'true_entropies': true_entropies,
            'mean_predicted_entropies': np.mean(predicted_entropies, axis=0),
            'std_predicted_entropies': np.std(predicted_entropies, axis=0),
            'model_config': model_config,
            'dataset_type': dataset_type
        }
    
    def save_results_summary(self, all_results: List[Dict[str, Any]]) -> None:
        """Save a summary of all evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.config.results_dir, f"results_summary_{timestamp}.txt")
        
        with open(summary_file, 'w') as f:
            f.write("ICL Experiment Results Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Experiment timestamp: {timestamp}\n")
            f.write(f"Context length: {self.config.context_length}\n")
            f.write(f"Vocab size: {self.config.vocab_size}\n")
            f.write(f"Mixing fraction: {self.config.mixing_fraction}\n\n")
            
            for result in all_results:
                model_config = result['model_config']
                dataset_type = result['dataset_type']
                mean_ents = result['mean_predicted_entropies']
                
                f.write(f"Model: depth={model_config[0]}, heads={model_config[1]}\n")
                f.write(f"Dataset: {dataset_type}\n")
                f.write(f"Mean predicted entropies: {mean_ents}\n")
                f.write(f"Std predicted entropies: {result['std_predicted_entropies']}\n")
                f.write(f"True entropies: {result['true_entropies']}\n")
                f.write("-" * 30 + "\n\n")
        
        print(f"Results summary saved to: {summary_file}")
