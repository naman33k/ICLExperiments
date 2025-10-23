"""
Configuration module for ICL experiments.
Contains all experiment parameters and settings.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ExperimentConfig:
    """Configuration class for ICL experiments."""
    
    # Dataset parameters
    context_length: int = 50
    vocab_size: int = 50
    mixing_fraction: float = 0.9  # 1 - p where p=0.1
    perm_or_random: str = 'perm'
    
    # Model parameters
    depths: List[int] = None
    heads: List[int] = None
    n_embd: int = 192
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 5e-5
    max_iters: int = 30000
    num_workers: int = 0
    
    # Evaluation parameters
    num_test_samples: int = 100
    
    # Results and output
    results_dir: str = "results"
    save_plots: bool = True
    
    # Random seed
    seed: int = 3407
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.depths is None:
            self.depths = [6, 9, 12]
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
