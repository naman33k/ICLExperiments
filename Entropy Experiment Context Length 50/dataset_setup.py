"""
Dataset setup and loading module for ICL experiments.
Handles creation and loading of different dataset types.
"""

import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import math
from typing import Dict, Tuple, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import InContextLearningExperiments.icldatasets_new_idea as datasets
from config import ExperimentConfig


class DatasetManager:
    """Manages dataset creation and loading for ICL experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.mixed_loaders = {}
        self.train_dataset_mixed_dict = {}
        
    def setup_mixed_dataset(self) -> Tuple[DataLoader, Any]:
        """Set up the mixed dictionary dataset for training."""
        l = self.config.context_length
        vocab_size = self.config.vocab_size
        
        # Create mixed dataset
        self.train_dataset_mixed_dict[l] = datasets.KVRetrievalDatasetMixedDictNewIdea(
            'train', 
            vocab_size=vocab_size, 
            length=l, 
            mixing_fraction=self.config.mixing_fraction, 
            perm_or_random=self.config.perm_or_random
        )
        
        # Create data loader
        self.mixed_loaders[l] = DataLoader(
            self.train_dataset_mixed_dict[l], 
            batch_size=self.config.batch_size, 
            num_workers=self.config.num_workers, 
            drop_last=False
        )
        
        return self.mixed_loaders[l], self.train_dataset_mixed_dict[l]
    
    def create_fixed_dataset_loader(self, dictionary: Any) -> DataLoader:
        """Create a fixed dictionary dataset loader for evaluation."""
        train_dataset_fixed_dict = datasets.KVRetrievalDatasetFixedDictNewIdea(
            'train', 
            vocab_size=self.config.vocab_size, 
            length=self.config.context_length, 
            mixing_fraction=1.0, 
            perm_or_random=self.config.perm_or_random, 
            dictionary=dictionary
        )
        
        return DataLoader(
            train_dataset_fixed_dict, 
            batch_size=1, 
            num_workers=0, 
            drop_last=False
        )
    
    def create_changing_dataset_loader(self) -> DataLoader:
        """Create a changing dictionary dataset loader for evaluation."""
        train_dataset_changing_dict = datasets.KVRetrievalDatasetChangingDictNewIdea(
            'train', 
            vocab_size=self.config.vocab_size, 
            length=self.config.context_length, 
            perm_or_random=self.config.perm_or_random
        )
        
        return DataLoader(
            train_dataset_changing_dict, 
            batch_size=1, 
            num_workers=0, 
            drop_last=False
        )
    
    def collect_samples(self, loader: DataLoader, num_samples: int = None) -> list:
        """Collect samples from a data loader."""
        if num_samples is None:
            num_samples = self.config.num_test_samples
            
        samples = []
        for j in range(num_samples):
            a, b = next(iter(loader))
            samples.append([a, b])
        return samples
    
    def collect_changing_samples(self, loader: DataLoader, mixed_loader: DataLoader, num_samples: int = None) -> list:
        """Collect samples from changing dataset that differ from mixed dataset."""
        if num_samples is None:
            num_samples = self.config.num_test_samples
            
        samples = []
        j = 0
        while j < num_samples:
            a, b = next(iter(loader))
            # Only include samples where the first element differs from mixed dataset
            if mixed_loader.dataset.dictionary[a[0][0]] != b[0][0]:
                j += 1
                samples.append([a, b])
        return samples
    
    def calculate_theoretical_entropies(self) -> Tuple[float, float]:
        """Calculate ideal and dumb mode entropies for comparison."""
        l = self.config.context_length
        p = 1 - self.config.mixing_fraction
        
        ideal = (np.log(float(math.factorial(l))) + p * (np.log(float(math.factorial(l))))) / (2 * l)
        dumb_mode = (np.log(float(math.factorial(l))) + p * l * (np.log(float(l)))) / (2 * l)
        
        return ideal, dumb_mode
