import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams.update({'font.size': 14})  # Increase base font size

def plot_review_experiments():
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Non-strategic experiment data (from llm_20241122_092709_results.json)
    conditions1 = ['Professional', 'Casual', '3-Sent', '2-Sent', '1-Sent', 'Base']
    mi_scores1 = [0.2783, 0.3083, 0.3642, 0.3067, 0.1617, -0.0042]
    judge_scores1 = [0.8300, 0.7000, 0.5067, 0.3300, 0.1733, 0.0000]

    # Strategic experiment data (from llm_20241122_091715_results.json)
    conditions2 = ['Topic\nChange', 'Related\nChange', '3-Sent', '2-Sent', '1-Sent', 'Base']
    mi_scores2 = [0.0033, 0.2108, 0.3042, 0.2567, 0.1467, 0.0042]
    judge_scores2 = [0.8000, 0.7700, 0.7850, 0.5533, 0.2533, 0.0000]

    # Plot settings
    width = 0.35
    x = np.arange(len(conditions1))

    # First subplot - Non-strategic
    bars1 = ax1.bar(x - width/2, mi_scores1, width, label='MI Score', color='skyblue')
    bars2 = ax1.bar(x + width/2, judge_scores1, width, label='Judge Score', color='lightcoral')

    ax1.set_ylabel('Score', fontsize=16)
    ax1.set_title('Non-Strategic Review Experiment', fontsize=16, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions1, rotation=45, ha='right')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Second subplot - Strategic
    bars3 = ax2.bar(x - width/2, mi_scores2, width, label='MI Score', color='skyblue')
    bars4 = ax2.bar(x + width/2, judge_scores2, width, label='Judge Score', color='lightcoral')

    ax2.set_ylabel('Score', fontsize=16)
    ax2.set_title('Strategic Review Experiment', fontsize=16, pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions2, rotation=45, ha='right')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()
    return fig

def plot_criteria_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))

    criteria = ['Correctness', 'Completeness', 'Conciseness']
    strategic_corr = [0.6455, 0.4541, -0.2201]
    nonstrategic_corr = [0.9036, 0.6824, -0.6825]

    x = np.arange(len(criteria))
    width = 0.35

    ax.bar(x - width/2, strategic_corr, width, label='Strategic', color='skyblue')
    ax.bar(x + width/2, nonstrategic_corr, width, label='Non-Strategic', color='lightcoral')

    ax.set_ylabel('Correlation with LLM Judge', fontsize=16)
    ax.set_title('TVD-MI vs LLM Judge Correlation by Criteria', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(criteria, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add significance markers
    significant_conditions = [(2, True), (0, False)]  # (index, is_strategic)
    for idx, is_strategic in significant_conditions:
        x_pos = idx - width/2 if is_strategic else idx + width/2
        ax.text(x_pos, 0.95, '*', ha='center', va='bottom', fontsize=16)

    plt.tight_layout()
    return fig

def plot_agreement_matrix():
    judges = ['Human', 'QWEN', 'GPT']
    criteria = ['Q1', 'Q2', 'Q3']

    data = np.array([
        [0.110, 0.081, 0.066],  # Human
        [0.020, 0.019, 0.003],  # QWEN
        [0.035, 0.045, 0.044]   # GPT
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt='.3f', 
                xticklabels=criteria, 
                yticklabels=judges,
                cmap='YlOrRd',
                annot_kws={"size": 12})

    ax.set_title('Human-LLM Agreement Patterns', fontsize=16, pad=20)
    ax.set_xlabel('Criteria', fontsize=14)
    ax.set_ylabel('Judge Type', fontsize=14)

    plt.tight_layout()
    return fig

# Generate all plots
if __name__ == "__main__":
    plot_review_experiments()
    plt.savefig('review_experiments.png', dpi=300, bbox_inches='tight')

    plot_criteria_comparison()
    plt.savefig('criteria_comparison.png', dpi=300, bbox_inches='tight')

    plot_agreement_matrix()
    plt.savefig('agreement_matrix.png', dpi=300, bbox_inches='tight')