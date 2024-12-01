import json
import numpy as np
from itertools import product
import pandas as pd

def load_data(model, n):
    """Load data for given model and question number"""
    filename = f"data/medical_judge_data/q{n}_experiment_data_{model}.json"
    with open(filename, 'r') as f:
        data = json.load(f)
    return [task["responses"] for task in data["tasks"]]

def calculate_information(responses1, responses2, metric='tvd'):
    """
    Calculate information metric between two response vectors

    Args:
        responses1, responses2: Response vectors to compare
        metric: 'tvd' for TVD only, 'mi' for mutual information only, 
               'tvd-mi' for combined metric
    """
    # Get joint distribution
    joint_counts = np.zeros((3,3))  # -1, 0, 1
    for r1, r2 in zip(responses1, responses2):
        i, j = r1+1, r2+1
        joint_counts[i,j] += 1

    joint_p = joint_counts / len(responses1)

    # Get marginals
    p1 = joint_p.sum(axis=1)
    p2 = joint_p.sum(axis=0)

    if metric == 'tvd':
        # Calculate independent distribution
        independent = np.outer(p1, p2)
        # Calculate TVD
        return np.abs(joint_p - independent).sum() / 2

    elif metric == 'mi':
        # Calculate mutual information
        mi = 0
        for i, j in product(range(3), range(3)):
            if joint_p[i,j] > 0:
                mi += joint_p[i,j] * np.log2(
                    joint_p[i,j] / (p1[i] * p2[j] + 1e-10)
                )
        return mi

    elif metric == 'tvd-mi':
        # Calculate both and multiply
        independent = np.outer(p1, p2)
        tvd = np.abs(joint_p - independent).sum() / 2

        mi = 0
        for i, j in product(range(3), range(3)):
            if joint_p[i,j] > 0:
                mi += joint_p[i,j] * np.log2(
                    joint_p[i,j] / (p1[i] * p2[j] + 1e-10)
                )
        return tvd * mi

    else:
        raise ValueError(f"Unknown metric: {metric}")

def verify_human_consistency(models, questions):
    """Verify human responses match across models for same question"""
    mismatches = []

    # Use first model as reference
    reference_model = models[0]

    for n in questions:
        reference_responses = load_data(reference_model, n)

        # Compare each other model against reference
        for model in models[1:]:
            responses = load_data(model, n)

            # First check if number of tasks match
            if len(responses) != len(reference_responses):
                print(f"WARNING: Different number of tasks for Q{n}")
                print(f"{reference_model} tasks: {len(reference_responses)}")
                print(f"{model} tasks: {len(responses)}")
                continue

            # Compare first 5 responses (humans) for each task
            for task_idx in range(len(reference_responses)):
                try:
                    human_resp1 = reference_responses[task_idx][:5]
                    human_resp2 = responses[task_idx][:5]

                    if human_resp1 != human_resp2:
                        mismatches.append({
                            'question': n,
                            'task': task_idx,
                            f'{reference_model}_responses': human_resp1,
                            f'{model}_responses': human_resp2
                        })
                except IndexError as e:
                    print(f"Error accessing responses for Q{n} task {task_idx}")
                    print(f"{reference_model} length: {len(reference_responses[task_idx]) if task_idx < len(reference_responses) else 'N/A'}")
                    print(f"{model} length: {len(responses[task_idx]) if task_idx < len(responses) else 'N/A'}")
                    raise e

    return mismatches

def analyze_all_data(models, metric='tvd'):
    """Generate full matrix and summary statistics using specified metric"""
    questions = [1, 2, 3]

    # Check consistency first
    mismatches = verify_human_consistency(models, questions)
    if mismatches:
        print("WARNING: Inconsistent human responses found:")
        for m in mismatches:
            print(f"Question {m['question']}, Task {m['task']}:")
            for model in models:
                print(f"{model}: {m[f'{model}_responses']}")
            print()
        raise ValueError("Human responses not consistent across models")

    # Load all response data
    all_responses = {}
    for model, n in product(models, questions):
        responses = load_data(model, n)
        # Convert to numpy array and transpose so each row is a judge
        responses = np.array(responses).T

        # Store responses for each human judge (5) and LLM
        for i in range(5):
            all_responses[f"Human{i+1}_Q{n}"] = responses[i]
        all_responses[f"{model.upper()}_Q{n}"] = responses[5]

    # Calculate information matrix
    labels = sorted(all_responses.keys())
    n = len(labels)
    info_matrix = np.zeros((n, n))

    for i, j in product(range(n), range(n)):
        if i != j:
            info = calculate_information(
                all_responses[labels[i]], 
                all_responses[labels[j]],
                metric=metric
            )
            info_matrix[i,j] = info

    df_full = pd.DataFrame(info_matrix, index=labels, columns=labels)
    df_summary = compute_aggregate_stats(df_full, [m.upper() for m in models])

    return df_full, df_summary

def compute_aggregate_stats(df_full, models=["QWEN", "GPT"]):
    """
    Compute aggregate statistics across all questions with flexible model combinations

    Args:
        df_full: Full DataFrame with TVD-MI scores
        models: List of model names in uppercase

    Returns:
        DataFrame with aggregate statistics
    """
    aggregates = {q: {} for q in [1,2,3]}

    for q in [1,2,3]:
        # Get label groups
        human_labels = [f'Human{i}_Q{q}' for i in range(1,6)]
        model_labels = [f'{model}_Q{q}' for model in models]
        all_labels = human_labels + model_labels

        # All pairs (excluding diagonal)
        all_pairs = []
        for l1, l2 in product(all_labels, all_labels):
            if l1 != l2:  # Exclude diagonal
                all_pairs.append(df_full.loc[l1,l2])

        # Store all-pairs average
        aggregates[q]['All'] = np.mean(all_pairs)

        # Human-Human pairs
        human_human = []
        for h1, h2 in product(human_labels, human_labels):
            if h1 != h2:
                human_human.append(df_full.loc[h1,h2])
        aggregates[q]['Human-Human'] = np.mean(human_human)

        # Human-Model pairs
        for model in models:
            model_label = f'{model}_Q{q}'
            human_model = []
            for h in human_labels:
                # Include both directions
                human_model.append(df_full.loc[h,model_label])
                human_model.append(df_full.loc[model_label,h])
            aggregates[q][f'Human-{model}'] = np.mean(human_model)

        # Model-Model pairs (if more than one model)
        if len(models) > 1:
            for m1, m2 in product(models, models):
                if m1 < m2:  # Only do pairs once (alphabetical order)
                    model_model = []
                    label1, label2 = f'{m1}_Q{q}', f'{m2}_Q{q}'
                    # Include both directions
                    model_model.append(df_full.loc[label1,label2])
                    model_model.append(df_full.loc[label2,label1])
                    aggregates[q][f'{m1}-{m2}'] = np.mean(model_model)

    return pd.DataFrame(aggregates).T

# Example usage:
models = ["qwen", "gpt"]  # Specify models once at top level
df_full, df_summary = analyze_all_data(models)
print("\nFull Matrix:")
print(df_full)
print("\nSummary Statistics:")
print(df_summary)