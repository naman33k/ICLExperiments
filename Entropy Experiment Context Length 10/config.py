"""
Configuration module for ICL experiments.
Contains all experiment parameters and settings.
"""

import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ExperimentConfig:
    """Configuration class for ICL experiments."""
    
    # Dataset parameters
    context_length: int = 10
    vocab_size: int = 10
    mixing_fraction: float = 0.9  # 1 - p where p=0.1
    perm_or_random: str = 'perm'
    
    # Model parameters
    depths: List[int] = None
    heads: List[int] = None
    n_embd: int = 192
    
    # Training parameters
    batch_size: int = 256
    learning_rate: float = 2e-4
    max_iters: int = 20000
    num_workers: int = 0
    
    # Evaluation parameters
    num_test_samples: int = 500
    
    # Results and output
    results_dir: str = "results"
    save_plots: bool = True
    
    # Random seed
    seed: int = 3407
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.depths is None:
            self.depths = [12]
        if self.heads is None:
            self.heads = [6]
        
        # Create results directory if it doesn't exist
        if self.save_plots:
            os.makedirs(self.results_dir, exist_ok=True)
    
    @property
    def block_size(self) -> int:
        """Calculate block size from context length."""
        return 2 * self.context_length - 1
    
    @property
    def model_vocab_size(self) -> int:
        """Calculate model vocab size."""
        return 2 * self.vocab_size
    
    def get_model_configs(self) -> List[Tuple[int, int]]:
        """Get all combinations of depth and head configurations."""
        import itertools
        return list(itertools.product(self.depths, self.heads))
    
    def get_dictionary(self, seed: int = None) -> torch.Tensor:
        """
        Generate a fixed dictionary based on the given seed.
        
        Args:
            seed: Random seed for dictionary generation. If None, uses self.seed.
            
        Returns:
            torch.Tensor: A fixed dictionary of size vocab_size with values in range [vocab_size, 2*vocab_size-1]
        """
        if seed is None:
            seed = self.seed
            
        # Set numpy random seed for reproducibility
        np.random.seed(seed)
        
        # Generate a fixed permutation of vocab_size elements
        # Values will be in range [vocab_size, 2*vocab_size-1]
        dictionary = torch.from_numpy(np.random.permutation(self.vocab_size) + self.vocab_size)
        
        return dictionary
