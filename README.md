## Papers

- [Class Project: Scalable Oversight through Information-Theoretic Evaluation](./papers/CS329H_Scalable_Oversight.pdf)

**Abstract:** This paper introduces a novel mechanism for scalable oversight, leveraging Total Variation Distance Mutual Information (TVD-MI) in a principal-agent framework. Our approach uniquely addresses the challenge of oversight with information asymmetry, where the principal lacks direct access to ground truth. Unlike classical methods requiring perfect probability estimates, our mechanism provides robust theoretical guarantees while remaining practically implementable. We prove that the mechanism is robust to specification gaming—neither principals nor agents gain significant utility from distorting their natural responses. We validate our theoretical results through comprehensive experiments in two high-stakes domains: scientific review and medical text assessment. Our experiments demonstrate that TVD-MI effectively detects strategic behavior in paper reviews and correlates more strongly with human agreement on correctness (0.110 ± 0.014) compared to LLM judges (0.020-0.035 ± 0.004). These results establish TVD-MI as a practical tool for scalable oversight while highlighting important limitations in current LLM-based evaluation approaches.

# Information-Theoretic LLM Evaluation Framework

This repository contains the implementation of an information-theoretic framework for evaluating Language Model (LLM) outputs, developed through two related works:

1. [Workshop Paper](https://openreview.net/forum?id=QqMnRGlRJk): "Implementability of Information Elicitation Mechanisms with Pre-Trained Language Models" presented at the [ICML TF2M workshop](https://icml.cc/virtual/2024/workshop/29956)
2. Class Project: "Information-Theoretic Measures for LLM Output Evaluation" (CS329 2023)

## Project Evolution

This work began as research into information-theoretic measures for LLM evaluation, first presented at [ICML TF2M workshop](https://icml.cc/virtual/2024/workshop/29956). The class project extends this foundation by:
- Implementing asynchronous API-based evaluation
- Adding comparative analysis between human and LLM judges
- Developing scalable oversight mechanisms for text quality

## Publications

### Original Workshop Paper
"Implementability of Information Elicitation Mechanisms with Pre-Trained Language Models"
- Develops theoretical foundations for information-theoretic LLM evaluation
- Introduces Difference of Entropies (DoE) estimator
- Provides initial empirical validation

### Class Project Extension
"Information-Theoretic Measures for LLM Output Evaluation"
- Implements practical framework for large-scale evaluation
- Compares human vs LLM judgment patterns
- Demonstrates applications in text quality assessment

## Citations

### Software
```bibtex
@software{robertson2023llm,
  author = {Robertson, Zachary and Bedi, Suhana and Lee, Hansol},
  title = {LLM Evaluation Framework},
  year = {2023},
  url = {https://github.com/zrobertson466920/CS329_Project},
  version = {1.0.0}
}
```

### Research Paper
```bibtex
@inproceedings{robertson2024implementability,
  title={Implementability of Information Elicitation Mechanisms with Pre-Trained Language Models},
  author={Robertson, Zachary and Cha, Hannah and Sheha, Andrew and Koyejo, Sanmi},
  booktitle={ICML 2024 Workshop on Theoretical Foundations of Foundation Models}
}
```

## Key Technical Components

This project implements a framework for evaluating and comparing Language Model (LLM) responses using information-theoretic measures and pairwise comparisons.

## Overview

The framework provides:
- Automated data generation from LLM responses
- Evaluation using both synthetic and LLM-based critics/judges
- Information-theoretic scoring mechanisms
- Asynchronous API handling for efficient processing
- Comprehensive tracking of API usage and mechanism calls

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a config.py file with your OpenAI API settings:
```python
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-4"  # or your preferred model
MAX_TOKENS = 4000
```

## Project Structure

```
.
├── async_llm_test.py     # Main experiment runner
├── api_utils.py          # API handling utilities
├── config.py            # Configuration settings
├── data/               # Data directory
│   └── ...            # Generated datasets
└── README.md
```

## Core Components

### ExperimentOracle Class
Manages experiment configuration and execution. Supports two modes:
- Synthetic: Uses predefined distributions for testing
- LLM: Uses actual LLM calls for evaluation

### API Utilities
Handles all LLM API interactions with:
- Asynchronous processing
- Rate limiting
- Usage tracking
- Error handling

### Evaluation Mechanisms
Implements two types of evaluators:
1. Critics: Assess information gain between responses
2. Judges: Perform pairwise comparisons of response quality

## Usage

### Basic Example

```python
import asyncio
from async_llm_test import ExperimentOracle

# Define experiment configuration
exp_config = {
    "exp_type": "llm",
    "num_agents": 3,
    "model_config": {
        "model_name": "gpt-4",
        "max_tokens": 4000,
        "temperature": 1.0
    },
    "task_description": "Abstract review task",
    "agent_perspectives": [
        {"strategy": "Please review the following abstract in three sentences."},
        {"strategy": "Please review the following abstract in three sentences."},
        {"strategy": None}  # Null model
    ],
    "data_config": {
        "n_tasks": 50,
        "preload": False
    }
}

# Create oracle and run experiment
async def run_experiment():
    oracle = ExperimentOracle(exp_config)
    await oracle.experiment()

if __name__ == "__main__":
    asyncio.run(run_experiment())
```

### Running Tests

```bash
python async_llm_test.py
```

## Output Format

The experiment generates two types of files:

1. Dataset Files (`*_data.json`):
```json
{
    "task_description": "...",
    "agent_perspectives": [...],
    "tasks": [
        {
            "context": "...",
            "responses": [...]
        }
    ],
    "metadata": {
        "model_config": {...},
        "data_config": {...},
        "generation_time": "..."
    }
}
```

2. Results Files (`*_results.json`):
```json
{
    "task_description": "...",
    "agent_perspectives": [...],
    "comparisons": [
        {
            "agent_pair": [i, j],
            "comparison_type": "critic|judge",
            "result": 0|1,
            "x": "...",
            "y": "...",
            "prompt": "..."
        }
    ]
}
```

## Statistics Tracking

The framework tracks:
- API calls and token usage
- Mechanism calls (critic and judge)
- Execution time and costs

Access statistics programmatically:
```python
from api_utils import get_api_stats, get_mechanism_stats

api_stats = get_api_stats()
mechanism_stats = get_mechanism_stats()
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request