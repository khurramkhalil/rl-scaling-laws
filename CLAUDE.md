# CLAUDE.md - AI Assistant Guidelines for rl-scaling-laws

## Project Overview

**rl-scaling-laws** is a research project focused on studying reinforcement learning scaling laws - investigating how RL systems scale with data, compute, and model size.

**Status:** Initial development phase - repository structure pending implementation.

## Repository Structure

```
rl-scaling-laws/
├── CLAUDE.md              # AI assistant guidelines (this file)
├── README.md              # Project documentation
├── .gitignore             # Git ignore patterns (to be added)
├── requirements.txt       # Python dependencies (to be added)
├── pyproject.toml         # Project configuration (to be added)
├── src/                   # Source code (to be added)
│   ├── __init__.py
│   ├── models/            # RL model implementations
│   ├── environments/      # Training environments
│   ├── training/          # Training loops and algorithms
│   ├── evaluation/        # Metrics and evaluation code
│   └── utils/             # Utility functions
├── experiments/           # Experiment configurations and scripts
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit and integration tests
├── data/                  # Data storage (gitignored)
├── results/               # Experiment results (gitignored)
└── configs/               # Configuration files
```

## Development Environment

### Expected Stack
- **Language:** Python 3.9+
- **ML Frameworks:** PyTorch, JAX, or similar
- **RL Libraries:** Gymnasium, Stable-Baselines3, RLlib, or custom implementations
- **Experiment Tracking:** Weights & Biases, MLflow, or TensorBoard
- **Testing:** pytest
- **Code Quality:** black, isort, flake8, mypy

### Setup (When Implemented)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Coding Conventions

### Python Style
- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use Google-style docstrings

### File Naming
- Use snake_case for Python files: `training_loop.py`
- Use snake_case for functions and variables: `compute_reward()`
- Use PascalCase for classes: `PolicyNetwork`
- Use SCREAMING_SNAKE_CASE for constants: `MAX_EPISODES`

### Import Organization
```python
# Standard library
import os
from typing import Dict, List, Optional

# Third-party
import numpy as np
import torch

# Local
from src.models import PolicyNetwork
from src.utils import set_seed
```

### Documentation
- All public functions must have docstrings
- Include type information in docstrings
- Document expected shapes for tensor arguments
- Add inline comments for complex logic

Example:
```python
def compute_returns(
    rewards: torch.Tensor,
    gamma: float = 0.99
) -> torch.Tensor:
    """Compute discounted returns from rewards.

    Args:
        rewards: Tensor of shape (batch_size, timesteps) containing rewards.
        gamma: Discount factor between 0 and 1.

    Returns:
        Tensor of shape (batch_size, timesteps) containing discounted returns.
    """
    ...
```

## Testing Guidelines

### Test Structure
```
tests/
├── conftest.py           # Shared fixtures
├── unit/                  # Unit tests
│   ├── test_models.py
│   └── test_utils.py
├── integration/           # Integration tests
│   └── test_training.py
└── fixtures/              # Test data and mocks
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py

# Run tests matching pattern
pytest -k "test_policy"
```

### Test Conventions
- Test files must start with `test_`
- Test functions must start with `test_`
- Use descriptive test names: `test_policy_network_forward_pass_correct_shape`
- Use fixtures for common setup
- Mock external dependencies (APIs, file I/O)

## Experiment Management

### Configuration
- Store experiment configs in YAML/JSON files
- Use hydra or similar for config management
- Track all hyperparameters

### Reproducibility
- Always set random seeds
- Log all dependencies and versions
- Save model checkpoints regularly
- Record git commit hash with experiments

### Naming Convention for Experiments
```
experiments/{experiment_type}/{date}_{description}/
```

Example: `experiments/ppo_scaling/2025-01-15_baseline_cartpole/`

## Git Workflow

### Branch Naming
- Features: `feature/{description}`
- Bugfixes: `fix/{description}`
- Experiments: `experiment/{description}`
- Claude sessions: `claude/{session-id}`

### Commit Messages
Follow conventional commits:
```
type(scope): description

[optional body]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `experiment`

Examples:
- `feat(models): add transformer-based policy network`
- `fix(training): correct gradient accumulation bug`
- `experiment(scaling): run compute scaling experiments`
- `docs: update README with installation instructions`

### Pull Request Guidelines
- Include description of changes
- Link related issues
- Add test coverage for new features
- Update documentation as needed
- Request review before merging

## Common Tasks

### Adding a New RL Algorithm
1. Create implementation in `src/training/`
2. Add configuration schema
3. Write unit tests
4. Add integration test with simple environment
5. Document hyperparameters and usage

### Running Experiments
```bash
# Example command structure (when implemented)
python -m src.train \
    --config configs/ppo_cartpole.yaml \
    --seed 42 \
    --wandb-project rl-scaling-laws
```

### Analyzing Results
- Use notebooks in `notebooks/` for analysis
- Save figures to `results/figures/`
- Export tables to `results/tables/`

## Security and Data Handling

### Sensitive Data
- Never commit API keys, tokens, or credentials
- Use environment variables for sensitive configuration
- Add sensitive files to `.gitignore`

### Data Storage
- Large datasets should be downloaded/generated, not committed
- Use `.gitignore` for `data/`, `results/`, and model checkpoints
- Document data sources and download procedures

## Troubleshooting

### Common Issues

**CUDA out of memory:**
- Reduce batch size
- Use gradient checkpointing
- Enable mixed precision training

**Reproducibility issues:**
- Ensure all seeds are set (Python, NumPy, PyTorch, CUDA)
- Check for non-deterministic operations
- Verify environment versions match

**Slow training:**
- Profile with PyTorch profiler
- Check data loading bottlenecks
- Verify GPU utilization

## Resources

### RL Scaling Laws Background
- [Scaling Laws for Deep Reinforcement Learning](https://arxiv.org/abs/2301.13442)
- [Chinchilla paper on compute-optimal training](https://arxiv.org/abs/2203.15556)

### RL Fundamentals
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [CleanRL implementations](https://github.com/vwxyzjn/cleanrl)

## AI Assistant Notes

### When Working on This Project

1. **Check current state first** - This is a new project; verify what has been implemented
2. **Follow conventions** - Use the coding style and patterns defined above
3. **Document thoroughly** - RL code especially needs clear documentation of shapes and algorithms
4. **Test rigorously** - RL can be sensitive to implementation details
5. **Track experiments** - Reproducibility is critical for scaling law research

### Key Priorities
- Reproducibility and experiment tracking
- Clean, well-documented implementations
- Comprehensive testing
- Clear configuration management

---

*Last updated: 2025-11-19*
