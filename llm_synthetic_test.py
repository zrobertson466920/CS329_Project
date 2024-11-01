#import asyncio
import numpy as np
from itertools import product
import random
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re

# Global request tracking
config_calls = 0
f_calls = 0
judge_calls = 0
config_tokens = 0
tokens = 0


class ExperimentOracle:
    def __init__(self, exp_type="synthetic", llm_model=None):
        """
        Initialize the ExperimentOracle with the specified experiment type.

        Parameters:
            - exp_type (str): 'synthetic' or 'llm'
            - llm_model: Placeholder for the LLM model (e.g., OpenAI's GPT)
        """
        self.exp_type = exp_type
        self.llm_model = llm_model  # Placeholder for actual LLM integration

        if self.exp_type == "synthetic":
            p_joint = np.array([[[0.3, 0.0], [0.0, 0.3]], [[0.05, 0.05], [0.2, 0.1]]])
            self.p_joint = p_joint
            self.num_agents = p_joint.ndim
            self.task_description = "The following are abstract reviews."
        elif self.exp_type == "llm":
            self.num_agents = 6
            self.task_description = "The following are abstract reviews."
        else:
            raise ValueError("exp_type must be either 'synthetic' or 'llm'.")

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

    def generate_llm_data(self, n_tasks, preload=True):
        """
        Generate data by making actual calls to an LLM or load preloaded data.

        Args:
        - n_tasks (int): Number of tasks to generate or load.
        - preload (bool): If True, load data from a preloaded file. If False, generate new data.

        Returns:
        - contexts (list): List of context strings.
        - agent_perspectives (list): List of agent perspective dictionaries.
        - string_data (list): List of dictionaries containing context and responses.
        """
        if preload:
            # Load preloaded data from JSON file
            file_name = "data/preload_llm_experiment_data.json"
            print(f"Loading preloaded data from {file_name}")

            with open(file_name, "r") as file:
                data = json.load(file)

            # Extract contexts, agent_perspectives, and string_data
            contexts = [task['context'] for task in data['tasks'][:n_tasks]]
            agent_perspectives = data['agent_perspectives']
            string_data = data['tasks'][:n_tasks]

            # Ensure we have the correct number of tasks
            if len(string_data) < n_tasks:
                print(f"Warning: Requested {n_tasks} tasks, but only {len(string_data)} available in preloaded data.")

        else:
            # Todo: Existing code for generating new data
            # Hard-coded file name
            file_name = "playground/peer_prediction/Assets/extracted_data_abstract_title.json"

            # Load data from JSON file
            print("Loading Data")
            with open(file_name, "r") as file:
                data = json.load(file)

            # Ensure we have enough contexts
            data = data[:n_tasks]
            contexts = [item["instruction"] for item in data]

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

        # Save the string data to JSON (for both preloaded and newly generated data)
        save_experiment_dataset(
            tasks=string_data,
            task_description=self.task_description,
            agent_perspectives=agent_perspectives,
            filename="data/llm_experiment_data.json",
        )

        return contexts, agent_perspectives, string_data

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


def experiment(oracle, n_tasks):
    """
    Run the experiment for different numbers of tasks and compare results.
    """
    if oracle.exp_type == "synthetic":
        true_tvd_mi_matrix = oracle.calculate_true_tvd_mi_matrix()

        # Print true TVD-MI matrix and agent scores
        print_matrix(true_tvd_mi_matrix, "True TVD-MI Scores")

        print("True Agent TVD-MI scores (sum over columns of matrix):")
        true_agent_scores = true_tvd_mi_matrix.sum(axis=0)
        for i, score in enumerate(true_agent_scores):
            print(f"Agent {i+1}: {score:.4f}")
        print()

    print("Generating Data\n")
    contexts, perspectives, responses = oracle.generate_data(n_tasks)

    print("Scoring Mechanism\n")
    all_comparisons = calculate_agent_scores(responses, oracle)

    print("Saving Results\n")
    save_experiment_results(
        task_description=oracle.task_description,
        agent_perspectives=perspectives,
        all_comparisons=all_comparisons,
        filename=f"data/{oracle.exp_type}_experiment_results.json",
    )

    return all_comparisons


def save_experiment_dataset(tasks, task_description, agent_perspectives, filename=None):
    """
    Saves the experiment dataset to a JSON file with the specified structure.

    Args:
        tasks (List[Dict[str, Any]]): A list of task dictionaries, each containing a context and its responses.
        task_description (str): A description of the task.
        agent_perspectives (List[Dict[str, Any]]): A list of agent perspective dictionaries.
        filename (Optional[str], optional): The filename to save the dataset. Defaults to None.

    Returns:
        str: The path to the saved JSON file.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"data/experiment_dataset_{timestamp}.json"
        )

    dataset = {
        "task_description": task_description,
        "agent_perspectives": agent_perspectives,
        "tasks": [
            {"context": task.get("context"), "responses": task["responses"]}
            for task in tasks
        ],
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


def main():
    """
    Main function to run the experiment.

    To switch between modes, change the `exp_type` parameter to 'synthetic' or 'llm'.
    """
    # Set experiment type: 'synthetic' or 'llm'
    exp_type = "synthetic"  # Change to 'llm' to use LLM mode
    oracle = ExperimentOracle(exp_type=exp_type)
    n_tasks = 50  # Updated number of tasks as per your request

    # Calculate and print the total number of calls before running the experiment
    f_total, judge_total = calculate_total_calls(oracle.get_num_agents(), n_tasks)
    print(f"For {n_tasks} tasks:")
    print(f"  Total f calls: {f_total}")
    print(f"  Total judge calls: {judge_total}")
    print(f"  Total oracle calls: {f_total + judge_total}")
    print()

    print("Starting Experiment\n")
    all_comparisons = experiment(oracle, n_tasks)

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


if __name__ == "__main__":
    main()
