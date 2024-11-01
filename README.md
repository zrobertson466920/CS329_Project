# LLM Evaluation Framework

This framework implements and compares two methods for evaluating Language Model (LLM) outputs:
1. Total Variation Distance Mutual Information (TVD-MI) 
2. Standard LLM-as-Judge approach

## Core Components

The framework consists of three main parts:

1. **ExperimentOracle** (`class ExperimentOracle`)
   - Handles data generation and evaluation logic
   - Supports both synthetic and LLM-based experiments
   - Methods for TVD-MI calculation and judge evaluation

2. **Mechanisms** (Multiple functions)
   - `generate_judge_prompt`: Creates prompts for LLM judge evaluation
   - `generate_tvd_mi_prompt`: Creates prompts for TVD-MI evaluation
   - `llm_approximation`: Approximates TVD-MI using LLM responses

3. **Experiment Runner** (`experiment()` function)
   - Coordinates experiment execution
   - Handles data collection and scoring
   - Saves results in standardized format

## Usage

### Basic Usage

```python
# Initialize oracle
oracle = ExperimentOracle(exp_type="llm")  # or "synthetic"
n_tasks = 50

# Run experiment
all_comparisons = experiment(oracle, n_tasks)
```

### Experiment Types

1. Synthetic Experiments
```python
oracle = ExperimentOracle(exp_type="synthetic")
```
- Uses predefined joint distribution
- Useful for validating implementation
- Ground truth available for comparison

2. LLM Experiments
```python
oracle = ExperimentOracle(exp_type="llm")
```
- Works with actual text responses
- Supports custom agent perspectives
- Handles real-world evaluation scenarios
- **TODOs:** upload your own data as argument, support LLM API calls

### Data Format

Input datasets should follow this structure:
```json
{
    "task_description": "string",
    "agent_perspectives": [
        {
            "reading": "string or null",
            "strategy": "string or null"
        }
    ],
    "tasks": [
        {
            "context": "string",
            "responses": ["string"]
        }
    ]
}
```

### Results Format

Results are saved in JSON format:
```json
{
    "task_description": "string",
    "agent_perspectives": [...],
    "comparisons": [
        {
            "agent_pair": [i, j],
            "result": float,
            "x": "string",
            "y": "string", 
            "comparison_type": "critic|judge",
            "prompt": "string"
        }
    ]
}
```

## Key Features

1. Efficient Subsampling
   - Reduces comparisons from O(t²) to O(t)
   - Maintains statistical validity
   
2. Symmetric Evaluation
   - Eliminates order bias
   - Balanced scoring across agents

3. Flexible Agent Perspectives
   - Supports multiple agent configurations
   - Handles information asymmetry scenarios

4. Results Analysis
   - TVD-MI matrix calculation
   - Judge score matrix
   - Per-agent performance metrics

## Resource Usage

The framework tracks:
- Number of LLM calls
- Token usage
- Execution time

Formula for total calls:
```python
f_total = num_agents * (num_agents - 1) * (2 * n_tasks)
judge_total = num_agents * (num_agents - 1) * (2 * n_tasks)
```
