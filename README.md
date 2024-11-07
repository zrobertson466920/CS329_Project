# Async LLM Testing Framework

This project implements an asynchronous framework for testing and evaluating LLM responses using peer prediction methods.

## Overview

The framework supports both synthetic and LLM-based experiments through a unified configuration system. It uses asynchronous API calls for efficient data generation and evaluation.

## Configuration

Experiments are configured using a dictionary structure:

```python
exp_config = {
    "exp_type": "llm" | "synthetic",  # Type of experiment
    "num_agents": int,                # Number of agents (3 for synthetic)
    "model_config": {
        "model_name": str,            # Name of LLM model
        "max_tokens": int,            # Max tokens per response
        "temperature": float,         # Temperature for generation
    },
    "agent_perspectives": [           # List of agent configurations
        {
            "reading": str | None,    # Reading transformation
            "strategy": str | None,   # Response strategy
        },
        ...
    ],
    "task_description": str,          # Description of the task
    "data_config": {
        "data_path": str,             # Path to input data
        "n_tasks": int,               # Number of tasks
        "preload": bool,              # Whether to use preloaded data
        "preload_path": str | None    # Path to preloaded data
    }
}
```

## Key Components

1. `ExperimentOracle`: Main class that handles experiment configuration and execution
2. Async Data Generation: Uses `asyncio` for efficient API calls
3. Data Saving: Automatic saving of experiment data and results
4. Metrics: Calculates TVD-MI scores and judge evaluations

## Usage

Basic usage:

```python
async def main():
    # Define experiment configuration
    exp_config = {...}  # Set configuration parameters
    
    # Create oracle
    oracle = ExperimentOracle(exp_config)
    
    # Run experiment
    all_comparisons = await experiment(oracle)
    
    # Results are automatically saved to data directory
```

## File Structure

- `async_llm_test.py`: Main implementation file
- `data/`: Directory for experiment data and results
  - Input data files
  - Generated experiment results
  - Debug output

## Requirements

- Python 3.12+
- OpenAI API access
- Required packages: numpy, asyncio, tiktoken

## Notes

- Synthetic mode currently only supports 3 agents
- LLM mode supports configurable number of agents
- All API calls are asynchronous for better performance
- Results are automatically saved with timestamps
