#import asyncio
import numpy as np
from itertools import product
import random
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
import asyncio
from openai import AsyncOpenAI
import tiktoken

try:
    from config import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS
except ImportError:
    print("Please create a config.py file with your API key and settings")
    raise

# Initialize OpenAI Async Client
client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
)

# Global request tracking
config_calls = 0
f_calls = 0
judge_calls = 0
config_tokens = 0
tokens = 0


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

        # Optional parameters with defaults
        self.model_config = exp_config.get("model_config", {
            "model_name": "gpt-4o-mini",
            "max_tokens": 4000,
            "temperature": 1.0
        })

        self.task_description = exp_config.get("task_description", 
            "The following are abstract reviews."
        )

        self.data_config = exp_config.get("data_config", {
            "data_path": "data/extracted_data_abstract_title.json",
            "n_tasks": 10,
            "preload": False,
            "preload_path": None
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

    def optimal_critic(self, index_pair, x, y):
        """
        Implement the optimal critic function.
        Translates string responses back to binary before computation.
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

    def optimal_judge(self, x, y):
        """
        Simulate an LLM judge for pairwise ranking.
        Returns 1 if x is preferred, 0 if y is preferred.
        Translates string responses back to binary before computation.

        Currently implemented as a dummy pass.

        To implement:
            - Replace the dummy logic with actual LLM-based judgment
        """
        x_bin = interpret_response_back(x)
        y_bin = interpret_response_back(y)

        global judge_calls
        judge_calls += 2
        # Dummy logic: prefer 'like' over 'dislike'
        return int(x_bin > y_bin)

    def llm_critic(self, index_pair, x, y):
        """
        Dummy placeholder
        Implement the optimal critic function.
        Translates string responses back to binary before computation.
        """
        if x == "This abstract discusses important research findings. The methodology appears sound. Further investigation may be warranted.":
            return 0
        elif y == "This abstract discusses important research findings. The methodology appears sound. Further investigation may be warranted.":
            return 1

        # Define a list of common stop words to ignore
        stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        # Function to extract keywords from text
        def extract_keywords(text):
            # Convert to lowercase and split into words
            words = re.findall(r'\w+', text.lower())
            # Remove stop words and keep words with length > 2
            return set(word for word in words if word not in stop_words and len(word) > 2)
        
        # Extract keywords from both texts
        x_keywords = extract_keywords(x)
        y_keywords = extract_keywords(y)
        
        # Calculate overlap
        overlap = len(x_keywords.intersection(y_keywords))
        total_keywords = len(x_keywords.union(y_keywords))
        
        # Calculate overlap ratio
        overlap_ratio = overlap / total_keywords if total_keywords > 0 else 0
        
        # Define threshold (you can adjust this value)
        threshold = 0.1
        
        # Return 1 if overlap ratio is above threshold, 0 otherwise
        return 1 if overlap_ratio > threshold else 0

    def llm_judge(self, x, y):
        """
        Dummy placeholder
        Simulate an LLM judge for pairwise ranking.
        Returns 1 if x is preferred, 0 if y is preferred.
        Translates string responses back to binary before computation.

        Currently implemented as a dummy pass.

        To implement:
            - Replace the dummy logic with actual LLM-based judgment
        """
        x_bin = len(x)
        y_bin = len(y)

        global judge_calls
        judge_calls += 1
        # Dummy logic: prefer 'like' over 'dislike'
        return int(x_bin > y_bin)

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

## API

def count_tokens(text: str, model = 'gpt-4o-mini') -> int:
    model = 'gpt-4o-mini'
    """
    Counts the number of tokens in the given text for the specified model.

    Args:
        text (str): The text to count tokens for.
        model (str): The OpenAI model name.

    Returns:
        int: The number of tokens.
    """
    encoding = tiktoken.encoding_for_model(model)

    return len(encoding.encode(text))

async def generate_completion_async(prompt: str, model_name: str, max_tokens: int) -> Tuple[Optional[str], int]:
    """
    Asynchronously generates a completion using OpenAI's Chat Completions API.

    Args:
        prompt (str): The prompt to send to the model.
        model_name (str): The name of the OpenAI model to use.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        Tuple[Optional[str], int]: A tuple containing the generated completion text (or None if an error occurs) and the number of tokens used in this call.
    """
    try:
        # Count tokens in the prompt
        token_count_prompt = count_tokens(prompt, model_name)

        # Create the chat completion asynchronously
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=1.0,
        )

        # Extract the assistant's reply
        assistant_message = response.choices[0].message.content

        # Count tokens in the response
        token_count_response = count_tokens(assistant_message, model_name)

        # Calculate total tokens for this call
        tokens_used = token_count_prompt + token_count_response

        return assistant_message, tokens_used

    except Exception as e:
        print(f"Error generating completion: {str(e)}")
        return None, 0

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
    global f_calls
    t = len(data)
    p_comparisons = []
    q_comparisons = []

    # p distribution
    for x, y in data:
        result = f(x, y)
        p_comparisons.append({"result": result, "distribution": "p", "x": x, "y": y})
    f_calls += t

    # q distribution (subsample t pairs instead of t*t)
    shuffled_pairs = list(product(*zip(*data)))
    random.shuffle(shuffled_pairs)
    subsampled_pairs = shuffled_pairs[:t]

    for x, y in subsampled_pairs:
        result = f(x, y)
        q_comparisons.append({"result": result, "distribution": "q", "x": x, "y": y})
    f_calls += t

    return p_comparisons + q_comparisons


def calculate_agent_scores(data, oracle):
    num_agents = oracle.get_num_agents()
    all_comparisons = []  # List to hold all comparisons

    for i in range(num_agents):
        print(f"Scoring Agent {i+1}")
        for j in range(num_agents):
            if i != j:
                # Define critic function that handles string data
                if oracle.exp_type == "synthetic":
                    f = lambda x, y: oracle.optimal_critic((i, j), x, y)
                elif oracle.exp_type == "llm":
                    f = lambda x, y: oracle.llm_critic((i, j), x, y)

                # Extract string responses for agent i and j
                pairwise_data = [(d["responses"][i], d["responses"][j]) for d in data]

                # Calculate TVD-MI score using approximation and get individual comparisons
                comparisons = llm_approximation(pairwise_data, f)

                # Add agent pair information and prompt to each comparison
                for comp in comparisons:
                    comp["agent_pair"] = (i + 1, j + 1)  # Agents are 1-indexed
                    comp["comparison_type"] = "critic"
                    comp["prompt"] = generate_tvd_mi_prompt(
                        oracle.task_description, comp["x"], comp["y"]
                    )

                all_comparisons.extend(comparisons)

                # Calculate LLM judge scores
                if oracle.exp_type == "synthetic":
                    f_judge = lambda x, y: oracle.optimal_judge(x, y)
                elif oracle.exp_type == "llm":
                    f_judge = lambda x, y: oracle.llm_judge(x, y)

                judge_comparisons = []
                for idx, (x, y) in enumerate(pairwise_data):
                    result = f_judge(x, y)
                    judge_comparisons.append(
                        {
                            "agent_pair": (i + 1, j + 1),
                            "result": result,
                            "x": x,
                            "y": y,
                            "comparison_type": "judge",
                            "prompt": generate_judge_prompt(oracle.task_description, x, y),
                        }
                    )

                all_comparisons.extend(judge_comparisons)
                global judge_calls
                judge_calls += len(pairwise_data)

    print()
    return all_comparisons

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
    Calculate the empirical TVD-MI matrix from individual comparisons.
    """
    empirical_tvd_mi = np.zeros((num_agents, num_agents))

    for i in range(1, num_agents + 1):
        for j in range(1, num_agents + 1):
            if i != j:
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
                    empirical_tvd_mi[i - 1][j - 1] = np.mean(p_results) - np.mean(
                        q_results
                    )

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
    Run the experiment with enhanced data saving.
    """

    if n_tasks is None:
        n_tasks = oracle.data_config["n_tasks"]

    start_time = datetime.now()
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

    if oracle.exp_type == "synthetic":
        true_tvd_mi_matrix = oracle.calculate_true_tvd_mi_matrix()
        print_matrix(true_tvd_mi_matrix, "True TVD-MI Scores")
        contexts, perspectives, responses = oracle.generate_data(n_tasks)
    else:
        print("Generating Data\n")
        contexts, perspectives, responses = await oracle.generate_llm_data(n_tasks)

    print("Scoring Mechanism\n")
    all_comparisons = calculate_agent_scores(responses, oracle)

    # Update metadata with final statistics
    end_time = datetime.now()
    metadata.update({
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "f_calls": f_calls,
        "judge_calls": judge_calls,
        "token_usage": {
            "total_tokens": tokens,
            "completion_tokens": config_tokens,
        }
    })

    # Save both the dataset and results
    save_experiment_dataset(
        tasks=responses,
        task_description=oracle.task_description,
        agent_perspectives=perspectives,
        filename=f"data/{oracle.exp_type}_experiment_data_{start_time:%Y%m%d_%H%M%S}.json",
        metadata=metadata
    )

    save_experiment_results(
        task_description=oracle.task_description,
        agent_perspectives=perspectives,
        all_comparisons=all_comparisons,
        filename=f"data/{oracle.exp_type}_experiment_results_{start_time:%Y%m%d_%H%M%S}.json"
    )

    return all_comparisons

async def main_async():
    """
    Main function to run the experiment.
    """
    # Define experiment configuration
    exp_config = {
        "exp_type": "llm",
        "num_agents": 3,
        "model_config": {
            "model_name": OPENAI_MODEL,
            "max_tokens": MAX_TOKENS,
            "temperature": 1.0
        },
        "agent_perspectives": [
            {"reading": None, "strategy": "Please review the following abstract in three sentences."},
            {"reading": None, "strategy": "Please review the following abstract in three sentences."},
            {"reading": None, "strategy": None}  # Null model
        ],
        "data_config": {
            "n_tasks": 10,
            "preload": False
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

    # Print the actual number of calls made during the experiment
    print(f"\nActual f calls: {f_calls}")
    print(f"Actual judge calls: {judge_calls}")
    print(f"Total actual oracle calls: {f_calls + judge_calls}")

def main():
    """
    Main function to run the experiment, calculate scores, and save results.
    """
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
