"""
Model training module for ICL experiments.
Handles GPT model creation, training, and management.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from typing import Dict, Tuple, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minGPT
from minGPT.mingpt.model import GPT
from minGPT.mingpt.trainer import Trainer
from minGPT.mingpt.utils import set_seed
from config import ExperimentConfig


class ModelManager:
    """Manages GPT model creation and training for ICL experiments with DDP support."""
    
    def __init__(self, config: ExperimentConfig, use_ddp: bool = True):
        self.config = config
        self.model_dicts = {}
        self.trainer_dicts = {}
        self.losses = []
        self.use_ddp = use_ddp
        
        # Initialize DDP if requested and available
        if self.use_ddp and torch.cuda.device_count() > 1:
            self.init_ddp()
        else:
            self.use_ddp = False
            print(f"Using single GPU training (devices available: {torch.cuda.device_count()})")
        
        # Set random seed
        set_seed(config.seed)
    
    def init_ddp(self):
        """Initialize Distributed Data Parallel."""
        if not dist.is_initialized():
            # Initialize the process group
            dist.init_process_group(backend='nccl')
        
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Set device for this process
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        print(f"DDP initialized: rank {self.rank}/{self.world_size}, local_rank {self.local_rank}")
    
    def cleanup_ddp(self):
        """Clean up DDP resources."""
        if self.use_ddp and dist.is_initialized():
            dist.destroy_process_group()
        
    def create_batch_end_callback(self):
        """Create callback function for training monitoring with DDP support."""
        def batch_end_callback(trainer):
            self.losses.append(trainer.loss.item())
            # Only print from rank 0 in DDP mode
            if trainer.iter_num % 100 == 0 and (not self.use_ddp or self.rank == 0):
                print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}; mean loss last 20 iters: {np.mean(self.losses[-20:]):.5f}")
                print(f"Current LR {trainer.optimizer.param_groups[0]['lr']:.7f}")
        return batch_end_callback
    
    def create_model_config(self, depth: int, heads: int) -> nn.Module:
        """Create a GPT model with specified configuration and wrap with DDP if needed."""
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
        
        # Create the base model
        model = GPT(model_config)
        
        # Move to appropriate device
        if self.use_ddp:
            model = model.to(self.device)
            # Wrap with DDP
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            # Use default device (cuda:0 or cpu)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
        
        return model
    
    def create_trainer_config(self) -> Trainer:
        """Create trainer configuration with DDP support."""
        train_config = Trainer.get_default_config()
        train_config.learning_rate = self.config.learning_rate
        train_config.max_iters = self.config.max_iters
        train_config.num_workers = self.config.num_workers
        train_config.lr_schedule = None
        
        # Adjust batch size for DDP
        if self.use_ddp:
            # Scale learning rate by world size for DDP
            train_config.learning_rate *= self.world_size
            print(f"Scaled learning rate for DDP: {train_config.learning_rate}")
        
        return train_config
    
    def train_models(self, train_dataset) -> Dict[Tuple[int, int], Tuple[nn.Module, Trainer]]:
        """Train all model configurations with DDP support."""
        model_configs = self.config.get_model_configs()
        
        for depth, heads in model_configs:
            # Only print from rank 0 in DDP mode
            if not self.use_ddp or self.rank == 0:
                print(f"\nTraining model with depth={depth}, heads={heads}")
            
            # Create model
            model = self.create_model_config(depth, heads)
            
            # Get parameter count (handle DDP wrapper)
            if self.use_ddp and hasattr(model, 'module'):
                param_count = sum(p.numel() for p in model.module.parameters())
            else:
                param_count = sum(p.numel() for p in model.parameters())
            
            if not self.use_ddp or self.rank == 0:
                print(f"Number of parameters: {param_count/1e6:.2f}M")
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
    
    def get_model(self, depth: int, heads: int) -> Tuple[nn.Module, Trainer]:
        """Get trained model and trainer for given configuration."""
        return self.model_dicts[(depth, heads)], self.trainer_dicts[(depth, heads)]
    
    def get_all_models(self) -> Dict[Tuple[int, int], Tuple[nn.Module, Trainer]]:
        """Get all trained models and trainers."""
        return {key: (self.model_dicts[key], self.trainer_dicts[key]) 
                for key in self.model_dicts.keys()}
    
    def unwrap_ddp_model(self, model: nn.Module) -> nn.Module:
        """Unwrap DDP model to get the underlying model."""
        if self.use_ddp and hasattr(model, 'module'):
            return model.module
        return model


def launch_ddp_training(config: ExperimentConfig, train_dataset, num_gpus: int = None):
    """
    Launch DDP training across multiple GPUs.
    
    Args:
        config: Experiment configuration
        train_dataset: Training dataset
        num_gpus: Number of GPUs to use (default: all available)
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 1:
        print("Single GPU detected, using regular training")
        model_manager = ModelManager(config, use_ddp=False)
        return model_manager.train_models(train_dataset)
    
    # Launch DDP training
    import subprocess
    import sys
    
    # Create launch command
    launch_cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        f"--nproc_per_node={num_gpus}",
        "--use_env",
        "run_icl_experiment_ddp.py"
    ]
    
    print(f"Launching DDP training with {num_gpus} GPUs")
    print(f"Command: {' '.join(launch_cmd)}")
    
    # Run the command
    result = subprocess.run(launch_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"DDP training failed with return code {result.returncode}")
        print(f"Error output: {result.stderr}")
        return None
    
    print("DDP training completed successfully")
    return result.stdout
