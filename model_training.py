"""
Model training module for ICL experiments.
Handles GPT model creation, training, and management.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List
import minGPT
from minGPT.mingpt.model import GPT
from minGPT.mingpt.trainer import Trainer
from minGPT.mingpt.utils import set_seed
from config import ExperimentConfig


class ModelManager:
    """Manages GPT model creation and training for ICL experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model_dicts = {}
        self.trainer_dicts = {}
        self.losses = []
        
        # Set random seed
        set_seed(config.seed)
        
    def create_batch_end_callback(self):
        """Create callback function for training monitoring."""
        def batch_end_callback(trainer):
            self.losses.append(trainer.loss.item())
            if trainer.iter_num % 100 == 0:
                print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}; mean loss last 20 iters: {np.mean(self.losses[-20:]):.5f}")
                print(f"Current LR {trainer.optimizer.param_groups[0]['lr']:.7f}")
        return batch_end_callback
    
    def create_model_config(self, depth: int, heads: int) -> GPT:
        """Create a GPT model with specified configuration."""
        model_config = GPT.get_default_config()
        model_config.model_type = None
        model_config.vocab_size = self.config.model_vocab_size
        model_config.block_size = self.config.block_size
        model_config.n_layer = depth
        model_config.n_head = heads
        model_config.n_embd = self.config.n_embd
        model_config.embd_pdrop = 0.
        model_config.resid_pdrop = 0.
        model_config.attn_pdrop = 0.
        
        return GPT(model_config)
    
    def create_trainer_config(self) -> Trainer:
        """Create trainer configuration."""
        train_config = Trainer.get_default_config()
        train_config.learning_rate = self.config.learning_rate
        train_config.max_iters = self.config.max_iters
        train_config.num_workers = self.config.num_workers
        train_config.lr_schedule = None
        return train_config
    
    def train_models(self, train_dataset) -> Dict[Tuple[int, int], Tuple[GPT, Trainer]]:
        """Train all model configurations."""
        model_configs = self.config.get_model_configs()
        
        for depth, heads in model_configs:
            print(f"\nTraining model with depth={depth}, heads={heads}")
            
            # Create model
            model = self.create_model_config(depth, heads)
            print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
            print(f"Running on device {next(model.parameters()).device}")
            
            # Create trainer
            train_config = self.create_trainer_config()
            trainer = Trainer(train_config, model, train_dataset)
            
            # Set up callback
            trainer.set_callback('on_batch_end', self.create_batch_end_callback())
            
            # Train model
            trainer.run()
            
            # Store trained model and trainer
            self.model_dicts[(depth, heads)] = model
            self.trainer_dicts[(depth, heads)] = trainer
            
        return self.model_dicts, self.trainer_dicts
    
    def get_model(self, depth: int, heads: int) -> Tuple[GPT, Trainer]:
        """Get trained model and trainer for given configuration."""
        return self.model_dicts[(depth, heads)], self.trainer_dicts[(depth, heads)]
    
    def get_all_models(self) -> Dict[Tuple[int, int], Tuple[GPT, Trainer]]:
        """Get all trained models and trainers."""
        return {key: (self.model_dicts[key], self.trainer_dicts[key]) 
                for key in self.model_dicts.keys()}
