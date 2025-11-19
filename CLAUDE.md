# CLAUDE.md - AI Assistant Guidelines for rl-scaling-laws

## Project Overview

**rl-scaling-laws** is a research framework for studying reinforcement learning scaling laws - investigating how RL systems scale with data, compute, and model size (10M-1B parameters).

**Status:** Phase 1 infrastructure complete - ready for pilot experiments.

## Repository Structure

```
rl-scaling-laws/
├── CLAUDE.md                    # AI assistant guidelines (this file)
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore patterns
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project configuration
├── src/rl_scaling_laws/         # Main source code
│   ├── __init__.py
│   ├── train.py                 # Main training entry point
│   ├── models/                  # Scalable neural network architectures
│   │   ├── base.py              # BaseNetwork class
│   │   ├── registry.py          # Model registry
│   │   ├── builder.py           # NetworkBuilder utility
│   │   ├── mlp.py               # MLP and ScalableMLP
│   │   ├── cnn.py               # CNN and ScalableCNN
│   │   ├── transformer.py       # Transformer architectures
│   │   └── moe.py               # Mixture of Experts
│   ├── algorithms/              # RL algorithm implementations
│   │   ├── base.py              # BaseAlgorithm class
│   │   ├── registry.py          # Algorithm registry
│   │   ├── sac.py               # Soft Actor-Critic
│   │   ├── dqn.py               # Deep Q-Network
│   │   └── ppo.py               # Proximal Policy Optimization
│   ├── environments/            # Environment wrappers and factories
│   │   ├── factory.py           # make_env, make_vec_env
│   │   ├── wrappers.py          # FrameStack, Normalize, etc.
│   │   └── registry.py          # Environment configurations
│   ├── training/                # Training infrastructure
│   │   ├── trainer.py           # Main Trainer class
│   │   ├── buffer.py            # ReplayBuffer, RolloutBuffer
│   │   └── callbacks.py         # WandB, Checkpoint, Eval callbacks
│   ├── evaluation/              # Evaluation and metrics
│   │   ├── evaluator.py         # Evaluator class
│   │   ├── metrics.py           # Scaling metrics, effective rank
│   │   └── video.py             # Video recording
│   └── utils/                   # Utility functions
│       ├── seed.py              # set_seed for reproducibility
│       ├── logging.py           # Logging setup
│       ├── param_count.py       # Parameter counting
│       ├── device.py            # Device management
│       └── schedule.py          # Learning rate schedules
├── configs/                     # Hydra configuration files
│   ├── config.yaml              # Main config
│   ├── algorithms/              # SAC, DQN, PPO configs
│   ├── architectures/           # MLP, CNN, Transformer, MoE
│   ├── environments/            # Atari, DMC, MetaWorld, Procgen
│   └── sweeps/                  # Experiment sweep configs
├── scripts/                     # Utility scripts
│   └── submit_job.py            # SLURM job submission
├── tests/                       # Test suite
│   ├── conftest.py              # Pytest fixtures
│   └── unit/                    # Unit tests
├── experiments/                 # Experiment outputs
├── analysis/                    # Analysis notebooks and scripts
├── data/                        # Data storage (gitignored)
├── results/                     # Results (gitignored)
└── logs/                        # Training logs (gitignored)
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Install optional dependencies for specific environments
pip install -e ".[atari]"     # Atari games
pip install -e ".[dmc]"       # DeepMind Control
pip install -e ".[all]"       # All environments
pip install -e ".[dev]"       # Development tools
```

### Running Training

```bash
# Basic training with default config
python -m rl_scaling_laws.train

# With config overrides
python -m rl_scaling_laws.train \
    scaling.target_params=100000000 \
    algorithm=sac \
    environment=dmc \
    seed=42

# Run with WandB logging
python -m rl_scaling_laws.train \
    logging.use_wandb=true \
    experiment.project=rl-scaling-laws
```

### Running Tests

```bash
pytest                              # All tests
pytest tests/unit/test_models.py    # Specific file
pytest -k "test_sac"                # Pattern match
pytest --cov=src --cov-report=html  # With coverage
```

## Key Components

### Scalable Architectures

All architectures automatically scale to target parameter counts:

```python
from rl_scaling_laws.models import ScalableMLP, NetworkBuilder

# Direct usage
model = ScalableMLP(
    input_dim=10,
    output_dim=5,
    target_params=100_000_000,  # 100M parameters
)

# Via builder
builder = NetworkBuilder("transformer", target_params=1_000_000_000)
actor = builder.build_actor(obs_dim=100, action_dim=10)
```

**Available architectures:**
- `mlp` - Multi-layer perceptron
- `cnn` - Convolutional neural network (for images)
- `transformer` - Transformer encoder
- `moe` - Mixture of Experts

### RL Algorithms

```python
from rl_scaling_laws.algorithms import SAC, DQN, PPO

# SAC for continuous control
sac = SAC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    architecture="mlp",
    target_params=10_000_000,
    utd_ratio=4,  # Updates per environment step
)

# Collect and train
sac.collect_rollouts(env, n_steps=1000)
metrics = sac.train()
```

### Environment Support

- **Atari** - Discrete actions, image observations
- **DMC** - Continuous control, vector observations
- **MetaWorld** - Robotic manipulation
- **Procgen** - Procedural generation (generalization test)

```python
from rl_scaling_laws.environments import make_env

env = make_env("atari", "Breakout", seed=42)
env = make_env("dmc", "walker_walk", seed=42)
```

### Configuration System

Uses Hydra for hierarchical configuration:

```yaml
# configs/config.yaml
defaults:
  - algorithm: sac
  - architecture: mlp
  - environment: dmc

scaling:
  target_params: 100_000_000
  architecture_type: mlp

training:
  total_timesteps: 10_000_000
  eval_freq: 50_000
```

Override via command line:
```bash
python -m rl_scaling_laws.train scaling.target_params=1000000000
```

## Development Conventions

### Code Style
- PEP 8 with 100 character line limit
- Type hints for all function signatures
- Google-style docstrings
- Use `black`, `isort`, `flake8`, `mypy`

### Naming Conventions
- Files: `snake_case.py`
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`

### Import Organization
```python
# Standard library
import os
from typing import Dict, List

# Third-party
import numpy as np
import torch

# Local
from rl_scaling_laws.models import ScalableMLP
from rl_scaling_laws.utils import set_seed
```

## Experiment Workflow

### 1. Configure Experiment

Create or modify config files in `configs/`:
```yaml
# configs/sweeps/my_experiment.yaml
hydra:
  sweeper:
    params:
      scaling.target_params: 10_000_000, 100_000_000, 1_000_000_000
      seed: 1, 2, 3, 4, 5
```

### 2. Run Experiment

```bash
# Single run
python -m rl_scaling_laws.train experiment.name=my_experiment

# Sweep (with Hydra multirun)
python -m rl_scaling_laws.train -m \
    scaling.target_params=10000000,100000000 \
    seed=1,2,3
```

### 3. Monitor with WandB

```bash
python -m rl_scaling_laws.train logging.use_wandb=true
```

### 4. Submit to Cluster

```bash
python scripts/submit_job.py \
    --config "scaling.target_params=1000000000" \
    --job-name scale_1B
```

## Scaling Law Analysis

### Key Metrics

```python
from rl_scaling_laws.evaluation.metrics import (
    compute_scaling_exponent,
    compute_effective_rank,
    compute_td_error,
)

# Fit power law: performance = A * params^alpha
alpha, A, r_squared = compute_scaling_exponent(params, performance)

# Track representation quality
rank = compute_effective_rank(activations)
```

### Experiment Targets

- **Pilot (Phase 1)**: 3 scales (10M, 100M, 1B), 1 domain, 1 algorithm
- **Full sweep (Phase 2)**: 5 scales × 3 algorithms × 4 domains
- **Analysis (Phase 3)**: Scaling exponents, Pareto frontiers, mechanistic analysis

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
training.batch_size=128

# Use gradient checkpointing (for large models)
# Enable mixed precision
```

### Reproducibility Issues
```python
from rl_scaling_laws.utils import set_seed
set_seed(42, deterministic=True)  # Slower but deterministic
```

### Slow Training
- Check GPU utilization with `nvidia-smi`
- Profile with PyTorch profiler
- Reduce logging frequency
- Use `AsyncVectorEnv` for parallel environments

## Git Workflow

### Branch Naming
- `feature/{description}` - New features
- `fix/{description}` - Bug fixes
- `experiment/{description}` - Experiment runs
- `claude/{session-id}` - Claude Code sessions

### Commit Messages
```
feat(models): add transformer-based policy network
fix(sac): correct target network update
experiment(scaling): run 1B parameter sweep
docs: update CLAUDE.md with implemented structure
```

## AI Assistant Notes

### When Working on This Project

1. **Use Hydra configs** - Don't hardcode hyperparameters
2. **Scale networks properly** - Use `NetworkBuilder` with `target_params`
3. **Track everything** - Use WandB callbacks for all experiments
4. **Test implementations** - Run `pytest` before committing
5. **Document shapes** - Always document tensor dimensions

### Key Files for Common Tasks

- **Add new algorithm**: `src/rl_scaling_laws/algorithms/`
- **Add new architecture**: `src/rl_scaling_laws/models/`
- **Modify training loop**: `src/rl_scaling_laws/training/trainer.py`
- **Add environment**: `src/rl_scaling_laws/environments/factory.py`
- **Configure experiment**: `configs/`

### Testing Commands
```bash
# Quick smoke test
pytest tests/unit/test_models.py -x

# Full test suite
pytest -v

# Test specific algorithm
pytest -k "SAC" -v
```

---

*Last updated: 2025-11-19*
*Phase 1 infrastructure: Complete*
