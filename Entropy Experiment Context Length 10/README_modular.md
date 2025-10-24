# ICL Experiment - Modularized Code

This directory contains the modularized version of the ICL (In-Context Learning) experiment from the Jupyter notebook. The code has been organized into separate modules for better maintainability and reusability.

## File Structure

```
ICLExperiments/
├── Entropy Experiment Context Length 10/     # This experiment directory
│   ├── config.py                 # Configuration parameters
│   ├── dataset_setup.py          # Dataset creation and loading
│   ├── model_training.py         # GPT model creation and training
│   ├── evaluation.py             # Model evaluation and plotting
│   ├── run_icl_experiment.py     # Main script to run the experiment
│   ├── test_modular.py           # Test script to verify modularization
│   ├── results/                  # Directory for saved plots and results
│   └── README_modular.md         # This file
├── OldNotebooksCheckpoint/       # Original Jupyter notebooks
├── ICLExperiments/              # Virtual environment
├── InContextLearningExperiments/ # Dataset modules
└── minGPT/                      # GPT implementation
```

## Modules Overview

### 1. `config.py`
- Contains `ExperimentConfig` class with all experiment parameters
- Centralized configuration management
- Easy to modify experiment settings

### 2. `dataset_setup.py`
- `DatasetManager` class handles dataset creation and loading
- Supports mixed, fixed, and changing dictionary datasets
- Calculates theoretical entropy bounds

### 3. `model_training.py`
- `ModelManager` class handles GPT model creation and training
- Supports multiple model configurations (different depths/heads)
- **Distributed Data Parallel (DDP) support** for multi-GPU training
- Automatic GPU detection and DDP initialization
- Training monitoring and callback management

### 4. `evaluation.py`
- `Evaluator` class handles model evaluation
- Calculates predicted vs true entropies
- Creates and saves plots with timestamps
- Generates results summary

### 5. `run_icl_experiment.py`
- Main script that orchestrates the entire experiment
- **Automatically detects multiple GPUs and uses DDP training**
- Runs dataset setup, model training, and evaluation
- Saves all results with timestamps

### 6. `run_icl_experiment_ddp.py`
- DDP-specific runner for distributed training
- Designed to be launched with `torch.distributed.launch`
- Handles multi-GPU training with proper synchronization

### 7. `test_modular.py`
- Test script to verify the modularization works correctly
- Tests imports, configuration, and basic functionality
- Run this first to ensure everything is set up properly

## Usage

### Prerequisites
1. Activate the virtual environment:
```bash
cd /home/naman33k/ICLExperiments
source ICLExperiments/bin/activate
```

2. Navigate to the experiment directory:
```bash
cd "Entropy Experiment Context Length 10"
```

### Basic Usage
```bash
# Run the full experiment (automatically detects and uses multiple GPUs)
python run_icl_experiment.py

# Or test the modularization first
python test_modular.py

# For explicit DDP training (if you have multiple GPUs)
python -m torch.distributed.launch --nproc_per_node=N run_icl_experiment_ddp.py
```

### Customizing Configuration
You can modify the experiment parameters by editing the `ExperimentConfig` class in `config.py`:

```python
@dataclass
class ExperimentConfig:
    context_length: int = 10        # Change context length
    vocab_size: int = 50           # Change vocabulary size
    depths: List[int] = [6, 9, 12] # Change model depths
    heads: List[int] = [6]         # Change number of attention heads
    max_iters: int = 30000         # Change training iterations
    # ... other parameters
```

### Using Individual Modules
You can also use individual modules in your own scripts:

```python
from config import ExperimentConfig
from dataset_setup import DatasetManager
from model_training import ModelManager
from evaluation import Evaluator

# Initialize with custom config
config = ExperimentConfig()
config.context_length = 25  # Custom setting

# Use individual components
dataset_manager = DatasetManager(config)
model_manager = ModelManager(config)
evaluator = Evaluator(config)
```

## Output

The experiment generates:

1. **Plots**: Saved in `results/` directory with timestamps
   - `entropy_comparison_fixed_depth{depth}_heads{heads}_{timestamp}.png`
   - `entropy_comparison_changing_depth{depth}_heads{heads}_{timestamp}.png`

2. **Results Summary**: Text file with all numerical results
   - `results_summary_{timestamp}.txt`

3. **Console Output**: Training progress and evaluation results

## Key Features

- **Modular Design**: Each component is separate and reusable
- **Distributed Data Parallel (DDP)**: Automatic multi-GPU training support
- **Timestamped Results**: All outputs include timestamps for easy tracking
- **Configurable**: Easy to modify experiment parameters
- **Error Handling**: Robust error handling and logging
- **Visualization**: High-quality plots with proper styling
- **Documentation**: Well-documented code with type hints
- **GPU Detection**: Automatically detects and utilizes available GPUs

## Dependencies

The code requires the same dependencies as the original notebook:
- torch
- numpy
- matplotlib
- seaborn
- scipy
- minGPT (local package)
- InContextLearningExperiments (local package)

## Notes

- The results directory is automatically created if it doesn't exist
- All plots are saved with high DPI (300) for publication quality
- The code maintains the same random seed (3407) for reproducibility
- CUDA is automatically detected and used if available
