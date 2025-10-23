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
            ans = model.generate(a.to(trainer.device), 1, do_sample=False)
            probs = F.softmax(ans[1], dim=-1)
            
            entropies = []
            for i in range(0, ans[1].size(1), 2):
                entropies.append(entropy(probs[0][i].cpu().numpy()))
            all_ents.append(entropies)
            
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
        
        # Calculate statistics
        stds = np.std(predicted_entropies, axis=0)
        mean_entropies = np.mean(predicted_entropies, axis=0)
        
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
        
        if dataset_type == 'fixed':
            plt.ylim(-0.3, 1.0)
        else:
            plt.ylim(-0.5, 3.0)
        
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
