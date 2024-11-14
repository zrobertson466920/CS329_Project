import numpy as np
from itertools import product
import random
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
import asyncio

from api_utils import (
    generate_completion_async,
    generate_batch_completions_async,
    count_tokens,
    get_api_stats,
    reset_api_stats,
    increment_f_calls,
    increment_judge_calls,  # Make sure this is imported
    get_mechanism_stats
)


try:
    from config import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS
except ImportError:
    print("Please create a config.py file with your API key and settings")
    raise


class ExperimentOracle:
    def __init__(self, exp_config: Dict[str, Any]):
        """
        Initialize ExperimentOracle with a configuration dictionary.

        Example exp_config:
        {
            "exp_type": "llm" | "synthetic",
            "num_agents": int,
            "model_config": {
                "model_name": str,
                "max_tokens": int,
                "temperature": float,
            },
            "agent_perspectives": [
                {
                    "reading": str | None,
                    "strategy": str | None
                },
                ...
            ],
            "task_description": str,
            "data_config": {
                "data_path": str,
                "n_tasks": int,
                "preload": bool,
                "preload_path": str | None
            }
        }
        """
        # Required parameters
        self.exp_type = exp_config["exp_type"]
        self.num_agents = exp_config["num_agents"]
        self.task_description = exp_config["task_description"]
        self.data_config = exp_config["data_config"]

        # Optional parameters with defaults
        self.model_config = exp_config.get("model_config", {
            "model_name": "gpt-4o-mini",
            "max_tokens": 4000,
            "temperature": 1.0
        })

        # Handle synthetic case
        if self.exp_type == "synthetic":
            if self.num_agents != 3:
                raise ValueError("Synthetic mode currently only supports 3 agents")
            self.p_joint = np.array([[[0.3, 0.0], [0.0, 0.3]], [[0.05, 0.05], [0.2, 0.1]]])

        # Set agent perspectives
        self.agent_perspectives = exp_config.get("agent_perspectives", [
            {"reading": None, "strategy": None} 
            for _ in range(self.num_agents)
        ])

    def generate_data(self, n_tasks):
        """
        Generate data based on the experiment type.

        Returns:
            - contexts of len n_tasks
            - agent_perspectives
            - string_data list of dicts (n_tasks, len(agent_perspectives))
        """
        if self.exp_type == "synthetic":
            return self.generate_synthetic_data(n_tasks)
        elif self.exp_type == "llm":
            return self.generate_llm_data(n_tasks)
        else:
            raise ValueError("Unknown experiment type.")

    def generate_synthetic_data(self, n_tasks):
        """
        Generate synthetic data from a JSON file and a list of agent perspectives.
        """

        # Generate dummy contexts
        contexts = [f"Synthetic context {i+1}" for i in range(n_tasks)]

        agent_perspectives = [
            {"reading": None, "strategy": None},
            {"reading": None, "strategy": None},
            {"reading": None, "strategy": None},
        ]

        # Generate responses for each context and perspective
        string_data = self.process_perspectives(agent_perspectives, contexts)

        # Add contexts to string_data
        for i, task in enumerate(string_data):
            task["context"] = contexts[i]

        # Save the string data to JSON
        save_experiment_dataset(
            tasks=string_data,
            task_description=self.task_description,
            agent_perspectives=agent_perspectives,
            filename="data/synthetic_experiment_data.json",
        )

        return contexts, agent_perspectives, string_data

    async def generate_llm_data(self, n_tasks=None, preload=None):
        """
        Generate data by making actual calls to an LLM or load preloaded data.

        Args:
            n_tasks (int, optional): Number of tasks to generate/load. Defaults to config value.
            preload (bool, optional): Whether to load preexisting data. Defaults to config value.

        Returns:
            Tuple[List[str], List[Dict], List[Dict]]: 
                - contexts: List of original contexts
                - agent_perspectives: List of agent perspective configurations
                - tasks: List of task dictionaries containing context and responses
        """
        # Use config values if not specified
        if n_tasks is None:
            n_tasks = self.data_config["n_tasks"]
        if preload is None:
            preload = self.data_config.get("preload", False)

        if preload:
            preload_path = self.data_config.get("preload_path", "data/preload_llm_experiment_data.json")
            print(f"Loading preloaded data from {preload_path}")

            with open(preload_path, "r") as file:
                data = json.load(file)

            # Extract contexts, agent_perspectives, and string_data
            contexts = [task['context'] for task in data['tasks'][:n_tasks]]
            agent_perspectives = data['agent_perspectives']
            tasks = data['tasks'][:n_tasks]

            # Ensure we have the correct number of tasks
            if len(tasks) < n_tasks:
                print(f"Warning: Requested {n_tasks} tasks, but only {len(tasks)} available in preloaded data.")

            return contexts, agent_perspectives, tasks

        # Load contexts from configured path
        data_path = self.data_config.get("data_path", "data/extracted_data_abstract_title.json")
        with open(data_path, "r") as file:
            data = json.load(file)
        contexts = [item["instruction"] for item in data[:n_tasks]]

        # Generate responses asynchronously 
        async def process_context(context):
            responses = []
            for perspective in self.agent_perspectives:
                if perspective["strategy"] is None:
                    # Null model constant response
                    response = "This abstract discusses important research findings. The methodology appears sound. Further investigation may be warranted."
                    tokens_used = count_tokens(response, self.model_config["model_name"])
                else:
                    prompt = f"{perspective.get('strategy', '')}\n\n{context}"
                    response, tokens_used = await generate_completion_async(
                        prompt,
                        self.model_config["model_name"],
                        self.model_config["max_tokens"]
                    )
                    response = response if response else "No response generated"

                responses.append(response)

            return {"context": context, "responses": responses}

        # Process all contexts in parallel
        print(f"Generating responses for {n_tasks} contexts...")
        tasks = await asyncio.gather(*[process_context(ctx) for ctx in contexts])

        # Save data immediately for debugging/inspection
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_filename = f"data/llm_data_debug_{timestamp}.json"

        save_experiment_dataset(
            tasks=tasks,
            task_description=self.task_description,
            agent_perspectives=self.agent_perspectives,
            filename=debug_filename,
            metadata={
                "model_config": self.model_config,
                "data_config": self.data_config,
                "generation_time": timestamp
            }
        )
        print(f"Debug data saved to {debug_filename}")

        return contexts, self.agent_perspectives, tasks

    def process_perspectives(self, perspectives, contexts):
        """
        Simulate a response based on the context and perspective.
        This is a placeholder and should be replaced with actual logic or LLM calls.
        """
        if self.exp_type == "synthetic":
            # Placeholder logic: randomly generate a positive or negative response
            flat_indices = np.random.choice(
                self.p_joint.size, size=len(contexts), p=self.p_joint.ravel()
            )
            binary_data = [
                np.unravel_index(idx, self.p_joint.shape) for idx in flat_indices
            ]
            # Convert binary responses to strings
            string_data = []
            for i, task in enumerate(binary_data):
                responses = [
                    f"I really {interpret_response(response)} it!" for response in task
                ]
                string_data.append({"context": contexts[i], "responses": responses})
            return string_data
        elif self.exp_type == "llm":
            # Dummy Code to be Replaced
            flat_indices = np.random.choice(
                self.p_joint.size, size=len(contexts), p=self.p_joint.ravel()
            )
            binary_data = [
                np.unravel_index(idx, self.p_joint.shape) for idx in flat_indices
            ]
            # Convert binary responses to strings
            string_data = []
            for i, task in enumerate(binary_data):
                responses = [
                    f"I really {interpret_response(response)} it!" for response in task
                ]
                string_data.append({"context": contexts[i], "responses": responses})
            return string_data

    async def get_critic(self, index_pair, x, y):
        if self.exp_type == "synthetic":
            score = self.preset_critic(index_pair, x, y)
            return score, "synthetic critic", {"type": "synthetic"}
        elif self.exp_type == "llm":
            return await self.llm_critic(x, y)
        else:
            score = self.simple_critic(index_pair, x, y)
            return score, "simple critic", {"type": "simple"}

    async def get_judge(self, x, y):
        if self.exp_type == "synthetic":
            score = self.preset_judge(x, y)
            return score, "synthetic judge", {"type": "synthetic"}
        elif self.exp_type == "llm":
            return await self.llm_judge(x, y)
        else:
            score = self.simple_judge(x, y)
            return score, "simple judge", {"type": "simple"}

    def preset_critic(self, index_pair, x, y):
        """
        Static critic using preset rules for testing without API calls.
        Used when exp_type is synthetic.
        """
        x_bin = interpret_response_back(x)
        y_bin = interpret_response_back(y)

        i, j = index_pair  # Assuming we're always comparing the first two dimensions
        pairwise_joint = self.get_pairwise_joint(i, j)
        p_marginal_1 = pairwise_joint.sum(axis=1)
        p_marginal_2 = pairwise_joint.sum(axis=0)
        return int(
            pairwise_joint[x_bin, y_bin] > p_marginal_1[x_bin] * p_marginal_2[y_bin]
        )

    def preset_judge(self, x, y):
        """
        Static judge using preset rules for testing without API calls.
        Used when exp_type is synthetic.
        """
        x_bin = interpret_response_back(x)
        y_bin = interpret_response_back(y)

        increment_judge_calls()  # Changed from global variable
        return int(x_bin > y_bin)

    def simple_critic(self, index_pair, x, y):
        """
        Binary critic function that returns 1 if responses are dependent, 0 if independent.
        """
        if x == "This abstract discusses important research findings. The methodology appears sound. Further investigation may be warranted.":
            return 0
        elif y == "This abstract discusses important research findings. The methodology appears sound. Further investigation may be warranted.":
            return 0

        # Define a list of common stop words to ignore
        stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])

        def extract_keywords(text):
            words = re.findall(r'\w+', text.lower())
            return set(word for word in words if word not in stop_words and len(word) > 2)

        x_keywords = extract_keywords(x)
        y_keywords = extract_keywords(y)

        # Calculate overlap
        overlap = len(x_keywords.intersection(y_keywords))
        total_keywords = len(x_keywords.union(y_keywords))

        # Calculate overlap ratio
        overlap_ratio = overlap / total_keywords if total_keywords > 0 else 0

        # Define threshold (you can adjust this value)
        threshold = 0.1

        # Return 1 if overlap ratio is above threshold (dependent), 0 otherwise (independent)
        return 1 if overlap_ratio > threshold else 0

    def simple_judge(self, x, y):
        """
        Basic text-based judge using length comparison.
        Used as fallback when API calls fail or for testing.
        """
        x_len = len(x)
        y_len = len(y)

        increment_judge_calls()  # Changed from global variable
        return int(x_len > y_len)

    async def llm_critic(self, response_a: str, response_b: str) -> tuple[float, str, dict]:
        """
        LLM-based critic that evaluates information gain between responses.
        Returns (score, raw_response, metadata)
        """
        increment_f_calls()
        prompt = generate_tvd_mi_prompt(self.task_description, response_a, response_b)

        response, metadata = await generate_completion_async(
            prompt=prompt,
            temperature=0.3
        )

        if response:
            score = interpret_tvd_mi_response(response)
            return score, response, metadata
        else:
            # Fallback to simple critic
            simple_score = self.simple_critic(None, response_a, response_b)
            return simple_score, "API call failed - using simple critic", metadata

    async def llm_judge(self, response_a: str, response_b: str) -> tuple[float, str, dict]:
        """
        LLM-based judge that compares response quality.
        Returns (score, raw_response, metadata)
        """
        increment_judge_calls()
        prompt = generate_judge_prompt(self.task_description, response_a, response_b)

        response, metadata = await generate_completion_async(
            prompt=prompt,
            temperature=0.3
        )

        if response:
            score = interpret_judge_response(response)
            return score, response, metadata
        else:
            # Fallback to simple judge
            simple_score = self.simple_judge(response_a, response_b)
            return simple_score, "API call failed - using simple judge", metadata

    def get_num_agents(self):
        return self.num_agents

    def calculate_true_tvd_mi_matrix(self):
        tvd_mi_matrix = np.zeros((self.num_agents, self.num_agents))

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    pairwise_joint = self.get_pairwise_joint(i, j)
                    tvd_mi_matrix[i, j] = self.true_tvd_mi(pairwise_joint)

        return tvd_mi_matrix

    def true_tvd_mi(self, pairwise_joint):
        """
        Calculate the true TVD-MI given the pairwise joint distribution.
        """
        p_marginal_1 = pairwise_joint.sum(axis=1)
        p_marginal_2 = pairwise_joint.sum(axis=0)
        return np.sum(np.abs(pairwise_joint - np.outer(p_marginal_1, p_marginal_2))) / 2

    def get_pairwise_joint(self, i, j):
        """
        Get the pairwise joint distribution for agents i and j.
        """
        pairwise_joint = self.p_joint.sum(
            axis=tuple(k for k in range(self.num_agents) if k not in (i, j))
        )
        return pairwise_joint / pairwise_joint.sum()  # Normalize


## Mechanisms


def generate_judge_prompt(query: str, response_a: str, response_b: str) -> str:
    prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie.

[User Question]
{query}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]
"""
    return prompt


def interpret_judge_response(response: str) -> float:
    if "[[A]]" in response:
        return 1.0
    elif "[[B]]" in response:
        return 0.0
    else:  # [[C]] for tie
        return 0.5


def interpret_tvd_mi_response(response: str) -> float:
    if "[[Significant Gain]]" in response:
        return 1.0
    elif "[[Little Gain]]" in response:
        return 0.25
    else:  # [[No Gain]]
        return 0.0


def generate_tvd_mi_prompt(
    task_description: str, response_a: str, response_b: str
) -> str:
    prompt_info_gain = f"""[System]
You are an impartial evaluator assessing the predictive information gain of one response vs. another. Your goal is to determine if knowing [Response A] provides more information than knowing [Response B]. Consider the following:

[Task Description]
{task_description}

[Response A]
{response_a}

[Response B]
{response_b}

Your task: Evaluate whether knowing [Response A] provides significantly more information than knowing just [Response B]. Consider unique details, complementary perspectives, or redundant information.

Provide a brief explanation of your reasoning, then output your final decision using one of these formats:
"[[Significant Gain]]" if knowing [Response A] provides a significant amount of information about [Response B]
"[[Little Gain]]" if knowing [Response A] provides little information about [Response B]
"[[No Gain]]" if knowing [Response A] provides no information about [Response B]
"""

    return prompt_info_gain


def interpret_response(response_val):
    """
    Interpret binary response values into descriptive strings.
    """
    return "like" if response_val == 1 else "dislike"


def interpret_response_back(response_str):
    """
    Convert descriptive string responses back to binary values.
    Correctly handles 'dislike' before 'like' to avoid substring issues.
    """
    response_str = response_str.lower()
    if "dislike" in response_str:
        return 0
    elif "like" in response_str:
        return 1
    else:
        raise ValueError(f"Unknown response string: {response_str}")


def llm_approximation(data, f):
    """
    Approximate TVD-MI using the LLM method for a pair of agents with subsampling.
    Now returns individual comparisons with pair information.
    """
    t = len(data)
    p_comparisons = []
    q_comparisons = []

    # p distribution
    for x, y in data:
        result = f(x, y)
        p_comparisons.append({"result": result, "distribution": "p", "x": x, "y": y})
    increment_f_calls(t)

    # q distribution (subsample t pairs instead of t*t)
    shuffled_pairs = list(product(*zip(*data)))
    random.shuffle(shuffled_pairs)
    subsampled_pairs = shuffled_pairs[:t]

    for x, y in subsampled_pairs:
        result = f(x, y)
        q_comparisons.append({"result": result, "distribution": "q", "x": x, "y": y})
    increment_f_calls(t)

    return p_comparisons + q_comparisons


async def calculate_agent_scores(data, oracle):
    num_agents = oracle.get_num_agents()
    all_comparisons = []

    # Increase batch size significantly
    BATCH_SIZE = 500  # 10x increase

    # Create all comparison tasks upfront
    comparison_tasks = []
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                # Get data pairs
                p_data = [(d["responses"][i], d["responses"][j]) for d in data]

                # Create shuffled q distribution
                shuffled_responses_j = [d["responses"][j] for d in data]
                random.shuffle(shuffled_responses_j)
                q_data = list(zip([d["responses"][i] for d in data], shuffled_responses_j))

                # Add all tasks to queue
                for x, y in p_data:
                    comparison_tasks.append({
                        "pair": (i, j),
                        "x": x,
                        "y": y,
                        "type": "critic",
                        "distribution": "p"
                    })
                for x, y in q_data:
                    comparison_tasks.append({
                        "pair": (i, j),
                        "x": x,
                        "y": y, 
                        "type": "critic",
                        "distribution": "q"
                    })
                for x, y in p_data:
                    comparison_tasks.append({
                        "pair": (i, j),
                        "x": x,
                        "y": y,
                        "type": "judge",
                        "distribution": None
                    })

    # Process in larger batches with rate limiting
    for i in range(0, len(comparison_tasks), BATCH_SIZE):
        batch = comparison_tasks[i:i + BATCH_SIZE]

        # Process batch concurrently
        results = await asyncio.gather(*[
            process_single_comparison(task, oracle)
            for task in batch
        ])

        all_comparisons.extend(results)

    return all_comparisons

async def process_single_comparison(task, oracle):
    """Process a single comparison task"""
    i, j = task["pair"]
    x, y = task["x"], task["y"]

    if task["type"] == "critic":
        result, raw_response, metadata = await oracle.get_critic((i, j), x, y)
        prompt = generate_tvd_mi_prompt(oracle.task_description, x, y)
        comparison_type = "critic"
    else:
        result, raw_response, metadata = await oracle.get_judge(x, y)
        prompt = generate_judge_prompt(oracle.task_description, x, y)
        comparison_type = "judge"

    return {
        "agent_pair": (i + 1, j + 1),
        "result": result,
        "x": x,
        "y": y,
        "comparison_type": comparison_type,
        "prompt": prompt,
        "distribution": task["distribution"],
        "raw_response": raw_response,
        "metadata": metadata
    }

def save_experiment_dataset(
    tasks: List[Dict[str, Any]], 
    task_description: str,
    agent_perspectives: List[Dict[str, Any]],
    filename: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Enhanced version of save_experiment_dataset that includes metadata.

    Args:
        tasks: List of task dictionaries containing context and responses
        task_description: Description of the task
        agent_perspectives: List of agent perspective dictionaries
        filename: Optional custom filename
        metadata: Optional metadata about the experiment run (tokens used, timing, etc)
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/experiment_dataset_{timestamp}.json"

    dataset = {
        "task_description": task_description,
        "agent_perspectives": agent_perspectives,
        "tasks": [{
            "context": task.get("context"),
            "responses": task["responses"]
        } for task in tasks],
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat()
    }

    try:
        with open(filename, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Experiment dataset saved to {filename}")
    except Exception as e:
        print(f"Failed to save experiment dataset: {e}")
        raise e

    return filename


def save_experiment_results(
    task_description: str,
    agent_perspectives: List[Dict[str, Any]],
    all_comparisons: List[Dict[str, Any]],
    filename: Optional[str] = None,
):
    """
    Saves the experiment results in the specified format to a JSON file.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_results_{timestamp}.json"

    experiment_results = {
        "task_description": task_description,
        "agent_perspectives": agent_perspectives,
        "comparisons": all_comparisons,
    }

    try:
        with open(filename, "w") as f:
            json.dump(experiment_results, f, indent=2)
        print(f"Experiment results saved to {filename}")
    except Exception as e:
        print(f"Failed to save experiment results: {e}")
        raise e


def calculate_total_calls(num_agents, n_tasks):
    """
    Calculate the total number of calls to f and judge functions.
    """
    f_total = (
        num_agents * (num_agents - 1) * (2 * n_tasks)
    )  # Changed from n_tasks + n_tasks * n_tasks
    judge_total = num_agents * (num_agents - 1) * (2 * n_tasks)
    return f_total, judge_total


def calculate_empirical_tvd_mi(all_comparisons, num_agents):
    """
    Calculate the empirical TVD-MI matrix using the binary critic function results.
    """
    empirical_tvd_mi = np.zeros((num_agents, num_agents))

    for i in range(1, num_agents + 1):
        for j in range(1, num_agents + 1):
            if i != j:
                # Get all critic results for this pair
                p_results = [
                    comp["result"]
                    for comp in all_comparisons
                    if comp["agent_pair"] == (i, j)
                    and comp["comparison_type"] == "critic"
                    and comp["distribution"] == "p"
                ]
                q_results = [
                    comp["result"]
                    for comp in all_comparisons
                    if comp["agent_pair"] == (i, j)
                    and comp["comparison_type"] == "critic"
                    and comp["distribution"] == "q"
                ]

                if p_results and q_results:
                    # Calculate means of binary results
                    p_mean = np.mean(p_results)  # P(f(X,Y)=1)
                    q_mean = np.mean(q_results)  # P(f(X',Y)=1)

                    # TVD-MI lower bound is difference between these probabilities
                    empirical_tvd_mi[i - 1][j - 1] = p_mean - q_mean

    return empirical_tvd_mi


# Function to print a matrix with headers
def print_matrix(matrix, title):
    num_agents = int(len(matrix[0]))
    print(f"{title} Matrix:")
    # Create header row
    header = "\t" + "\t".join([f"Agent {j}" for j in range(1, num_agents + 1)])
    print(header)
    # Print each row with agent label
    for i in range(num_agents):
        row = f"Agent {i+1}\t" + "\t".join(
            [f"{matrix[i][j]:.4f}" for j in range(num_agents)]
        )
        print(row)
    print()

async def experiment(oracle, n_tasks = None):
    """
    Run the experiment with data saved after generation to preserve results.
    """
    if n_tasks is None:
        n_tasks = oracle.data_config["n_tasks"]

    start_time = datetime.now()
    base_filename = f"data/{oracle.exp_type}_{start_time:%Y%m%d_%H%M%S}"

    metadata = {
        "experiment_type": oracle.exp_type,
        "n_tasks": n_tasks,
        "num_agents": oracle.get_num_agents(),
        "start_time": start_time.isoformat(),
        "token_usage": {
            "total_tokens": 0,
            "completion_tokens": 0,
            "prompt_tokens": 0
        },
        "api_calls": {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0
        }
    }

    # Generate data
    if oracle.exp_type == "synthetic":
        true_tvd_mi_matrix = oracle.calculate_true_tvd_mi_matrix()
        print_matrix(true_tvd_mi_matrix, "True TVD-MI Scores")
        contexts, perspectives, responses = oracle.generate_data(n_tasks)
    else:
        print("Generating Data\n")
        contexts, perspectives, responses = await oracle.generate_llm_data(n_tasks)

    # Save generated data immediately
    data_filename = f"{base_filename}_data.json"
    save_experiment_dataset(
        tasks=responses,
        task_description=oracle.task_description,
        agent_perspectives=perspectives,
        filename=data_filename,
        metadata=metadata
    )
    print(f"Generated data saved to {data_filename}")

    # Calculate scores - Add await here
    print("\nScoring Mechanism")
    all_comparisons = await calculate_agent_scores(responses, oracle)  # Added await

    # Save scoring results
    results_filename = f"{base_filename}_results.json"
    save_experiment_results(
        task_description=oracle.task_description,
        agent_perspectives=perspectives,
        all_comparisons=all_comparisons,
        filename=results_filename,
    )
    print(f"Results saved to {results_filename}")

    return all_comparisons

async def main_async():
    """
    Main function to run the experiment.
    """
    # Define experiment configuration
    exp_config = {
        "exp_type": "llm",
        "num_agents": 6,
        "model_config": {
            "model_name": OPENAI_MODEL,
            "max_tokens": MAX_TOKENS,
            "temperature": 1.0
        },
        "task_description": "The following are abstract reviews.",
        "agent_perspectives": [
            {"reading": None, "strategy": "Please review the following abstract in three sentences."},
            {"reading": None, "strategy": "Please review the following abstract in two sentences."},
            {"reading": None, "strategy": None}  # Null model
        ],
        "data_config": {
            "n_tasks": 10,
            "preload": True,
            "preload_path": "data/preload_llm_experiment_data.json"
        }
    }

    oracle = ExperimentOracle(exp_config)

    # Use n_tasks from config
    n_tasks = exp_config["data_config"]["n_tasks"]

    # Calculate and print the total number of calls before running the experiment
    f_total, judge_total = calculate_total_calls(oracle.get_num_agents(), n_tasks)
    print(f"For {n_tasks} tasks:")
    print(f"  Total f calls: {f_total}")
    print(f"  Total judge calls: {judge_total}")
    print(f"  Total oracle calls: {f_total + judge_total}")
    print()

    print("Starting Experiment\n")
    all_comparisons = await experiment(oracle, n_tasks)

    # Calculate empirical TVD-MI matrix
    num_agents = oracle.get_num_agents()
    empirical_tvd_mi_matrix = calculate_empirical_tvd_mi(all_comparisons, num_agents)

    # Print Empirical TVD-MI Scores Matrix
    print_matrix(empirical_tvd_mi_matrix, "Empirical TVD-MI Scores")

    # Calculate and print Judge Scores Matrix
    judge_matrix = np.zeros((num_agents, num_agents))
    for comp in all_comparisons:
        if comp["comparison_type"] == "judge":
            i, j = comp["agent_pair"]
            judge_matrix[i - 1][j - 1] += comp["result"]

    # Normalize judge scores
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                total_comparisons = sum(
                    1
                    for comp in all_comparisons
                    if comp["comparison_type"] == "judge"
                    and comp["agent_pair"] == (i + 1, j + 1)
                )
                if total_comparisons > 0:
                    judge_matrix[i][j] /= total_comparisons

    print_matrix(judge_matrix, "Judge Scores")

    # Calculate and print the summed scores for each agent
    mean_tvd_scores = empirical_tvd_mi_matrix.mean(axis=1)
    mean_judge_scores = judge_matrix.mean(axis=1)

    print(f"Objective Critic Score - {mean_tvd_scores.mean()}")

    # Print the summed scores
    print("Mean Scores per Agent:")
    for agent in range(1, num_agents + 1):
        print(
            f"  Agent {agent} - Total TVD-MI score: {mean_tvd_scores[agent-1]:.4f}, "
            f"Mean Judge score: {mean_judge_scores[agent-1]:.4f}"
        )
    print()

    print("\nMechanism Call Statistics:")
    print(f"  f_calls: {get_mechanism_stats()['f_calls']}")
    print(f"  judge_calls: {get_mechanism_stats()['judge_calls']}")
    print(f"  Total mechanism calls: {get_mechanism_stats()['f_calls'] + get_mechanism_stats()['judge_calls']}")

    print("\nAPI Statistics:")
    print(f"  Total API calls: {get_api_stats()['calls']}")
    print(f"  Total tokens: {get_api_stats()['tokens']['total']}")
    print(f"  Duration: {get_api_stats()['duration']:.2f} seconds")

def main():
    """
    Main function to run the experiment, calculate scores, and save results.
    """
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
