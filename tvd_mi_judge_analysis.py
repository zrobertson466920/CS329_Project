import json
import numpy as np
from itertools import product
import pandas as pd
from scipy.stats import norm

def load_data(model, n):
    """Load data for given model and question number"""
    filename = f"data/medical_judge_data/q{n}_experiment_data_{model}.json"
    with open(filename, 'r') as f:
        data = json.load(f)
    return [task["responses"] for task in data["tasks"]]

def load_aligned_data(models, n):
    """
    Load and align data across models for question n
    Returns only tasks that exist and match across all models
    """
    # Load all data first
    model_data = {}
    for model in models:
        filename = f"data/medical_judge_data/q{n}_experiment_data_{model}.json"
        with open(filename, 'r') as f:
            data = json.load(f)
        model_data[model] = data["tasks"]

    # Find minimum task count
    min_tasks = min(len(data) for data in model_data.values())

    # Track kept and dropped tasks
    kept_tasks = []
    dropped_tasks = {
        'missing': [],
        'mismatch_responses': [],
        'mismatch_context': []
    }

    # Check each task
    for task_idx in range(min_tasks):
        # Get human responses and contexts for each model
        human_responses = {}
        contexts = {}
        for model in models:
            if task_idx < len(model_data[model]):
                task = model_data[model][task_idx]
                human_responses[model] = task["responses"][:5]
                contexts[model] = task["context"]

        # Check if responses and contexts match across all models
        reference_resp = human_responses[models[0]]
        reference_context = contexts[models[0]]

        responses_match = all(human_responses.get(model, None) == reference_resp 
                            for model in models[1:])
        contexts_match = all(contexts.get(model, None) == reference_context 
                           for model in models[1:])

        if responses_match and contexts_match:
            # Everything matches, keep task
            kept_tasks.append(task_idx)
        else:
            if not responses_match:
                dropped_tasks['mismatch_responses'].append(task_idx)
            if not contexts_match:
                dropped_tasks['mismatch_context'].append(task_idx)

    print(f"\nQ{n} Alignment Summary:")
    print(f"- Original tasks: {len(model_data[models[0]])}")
    print(f"- Kept tasks: {len(kept_tasks)}")
    print(f"- Dropped due to response mismatches: {len(dropped_tasks['mismatch_responses'])}")
    print(f"- Dropped due to context mismatches: {len(dropped_tasks['mismatch_context'])}")

    # Print first mismatch example if any
    if dropped_tasks['mismatch_context']:
        first_idx = dropped_tasks['mismatch_context'][0]
        print(f"\nFirst Context Mismatch (Task {first_idx}):")
        for model in models:
            print(f"\n{model.upper()} Context Preview:")
            print(contexts[model][:200] + "...")

    # Return aligned data
    return {
        model: [model_data[model][i] for i in kept_tasks]
        for model in models
    }

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
    task_counts = {}

    # Get all task counts first
    print("\nTask Counts Summary:")
    for model in models:
        task_counts[model] = {}
        for n in questions:
            responses = load_data(model, n)
            task_counts[model][n] = len(responses)

    # Print task count differences
    ref_model = models[0]
    for n in questions:
        ref_count = task_counts[ref_model][n]
        for model in models[1:]:
            count = task_counts[model][n]
            if ref_count != count:
                print(f"Q{n}: {ref_model}={ref_count} vs {model}={count}")

    print("\nChecking Human Response Consistency...")
    # Check response consistency
    for n in questions:
        reference_responses = load_data(ref_model, n)

        for model in models[1:]:
            responses = load_data(model, n)
            min_tasks = min(len(responses), len(reference_responses))

            q_mismatches = []
            for task_idx in range(min_tasks):
                try:
                    human_resp1 = reference_responses[task_idx][:5]
                    human_resp2 = responses[task_idx][:5]

                    if human_resp1 != human_resp2:
                        q_mismatches.append(task_idx)
                        # Only store first mismatch details
                        if len(mismatches) == 0:
                            mismatches.append({
                                'question': n,
                                'task': task_idx,
                                'responses': {
                                    ref_model: human_resp1,
                                    model: human_resp2
                                }
                            })

                except IndexError as e:
                    print(f"Response length error Q{n} Task {task_idx}")
                    raise e

            if q_mismatches:
                print(f"Q{n}: Found {len(q_mismatches)} mismatches between {ref_model}-{model}")

    # Print first mismatch example if any found
    if mismatches:
        m = mismatches[0]
        print("\nFirst Mismatch Example:")
        print(f"Q{m['question']} Task {m['task']}:")
        for model, resp in m['responses'].items():
            print(f"{model}: {resp}")

    return mismatches, task_counts

def analyze_all_data(models, metric='tvd'):
    """Generate full matrix and summary statistics using specified metric"""
    questions = [1, 2, 3]

    # Load aligned data for each question
    all_responses = {}
    for n in questions:
        aligned_data = load_aligned_data(models, n)

        for model in models:
            # Convert to numpy array and transpose
            responses = np.array([task["responses"] for task in aligned_data[model]]).T

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
    Compute aggregate statistics with standard errors and significance tests
    """
    aggregates = {q: {} for q in [1,2,3]}
    std_errors = {q: {} for q in [1,2,3]}

    for q in [1,2,3]:
        # Get label groups
        human_labels = [f'Human{i}_Q{q}' for i in range(1,6)]
        model_labels = [f'{model}_Q{q}' for model in models]
        all_labels = human_labels + model_labels

        # Collect scores for each category
        all_scores = []
        human_scores = []
        model_scores = {model: [] for model in models}

        # All pairs
        for l1, l2 in product(all_labels, all_labels):
            if l1 != l2:
                score = df_full.loc[l1,l2]
                all_scores.append(score)

                # Human scores
                if 'Human' in l1:
                    human_scores.append(score)

                # Model scores
                for model in models:
                    if model in l1:
                        model_scores[model].append(score)

        # Calculate means and standard errors
        aggregates[q]['All'] = np.mean(all_scores)
        std_errors[q]['All'] = np.std(all_scores) / np.sqrt(len(all_scores))

        aggregates[q]['Human'] = np.mean(human_scores)
        std_errors[q]['Human'] = np.std(human_scores) / np.sqrt(len(human_scores))

        for model in models:
            aggregates[q][model] = np.mean(model_scores[model])
            std_errors[q][model] = np.std(model_scores[model]) / np.sqrt(len(model_scores[model]))

    # Create DataFrames
    df_means = pd.DataFrame(aggregates).T
    df_se = pd.DataFrame(std_errors).T

    # Format output with means ± SE
    df_formatted = pd.DataFrame(index=df_means.index, columns=df_means.columns)
    for col in df_means.columns:
        df_formatted[col] = df_means[col].map('{:.3f}'.format) + ' ± ' + df_se[col].map('{:.3f}'.format)

    return df_formatted, df_means, df_se  # Return raw values too for significance testing

def run_significance_tests(df_means, df_se):
    """Run significance tests between questions for each agent type"""
    tests = {}
    for col in df_means.columns:
        # Pairwise t-tests between questions
        pairs = [(1,2), (1,3), (2,3)]
        tests[col] = {}
        for q1, q2 in pairs:
            # Calculate t-statistic
            diff = df_means.loc[q1,col] - df_means.loc[q2,col]
            se_diff = np.sqrt(df_se.loc[q1,col]**2 + df_se.loc[q2,col]**2)
            t_stat = diff / se_diff
            # Two-tailed p-value (could use scipy.stats for more precise p-values)
            p_val = 2 * (1 - norm.cdf(abs(t_stat)))
            tests[col][f'Q{q1}-Q{q2}'] = p_val

    return pd.DataFrame(tests)

# Example usage:
models = ["QWEN", "GPT"]  # Specify models once at top level
df_full, df_summary = analyze_all_data(models)
# Usage:
df_formatted, df_means, df_se = compute_aggregate_stats(df_full, models)
significance = run_significance_tests(df_means, df_se)

print("\nTVD-MI Scores (mean ± SE):")
print(df_formatted)
print("\nSignificance Tests (p-values):")
print(significance)