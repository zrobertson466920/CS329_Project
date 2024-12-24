import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams.update({'font.size': 14})  # Increase base font size

def plot_review_experiments():
    # Create single figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Strategic experiment data
    conditions = ['Topic\nChange', 'Related\nChange', '3-Sent', '2-Sent', '1-Sent', 'Constant']
    mi_scores = [0.0033, 0.2108, 0.3042, 0.2567, 0.1467, 0.0042]
    judge_scores = [0.8000, 0.7700, 0.7850, 0.5533, 0.2533, 0.0000]

    # Plot settings
    width = 0.35
    x = np.arange(len(conditions))

    # Create bars
    bars1 = ax.bar(x - width/2, mi_scores, width, label='MI Score', color='#4E79A7')
    bars2 = ax.bar(x + width/2, judge_scores, width, label='Judge Score', color='#E15759')

    # Customize plot
    ax.set_ylabel('Score', fontsize=16)
    ax.set_title('Strategic Review Experiment', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

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

import matplotlib.pyplot as plt
import numpy as np

def plot_preference_distribution():
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # Create a more complex grid to allow for centered bottom plot
    gs = fig.add_gridspec(2, 4)  # 2 rows, 4 columns for finer control
    
    # Create axes with the new layout
    ax1 = fig.add_subplot(gs[0, 0:2])  # Top left, spans 2 columns
    ax2 = fig.add_subplot(gs[0, 2:4])  # Top right, spans 2 columns
    ax3 = fig.add_subplot(gs[1, 1:3])  # Bottom center, spans middle 2 columns
    
    # Data for each question
    q1_data = {
        'Prefers human': [24.2, 4.3, 0.0],
        'Neutral': [10.1, 2.5, 4.6],
        'Prefers gpt': [65.7, 93.1, 95.4]
    }
    
    q2_data = {
        'Prefers human': [11.6, 3.6, 0.0],
        'Neutral': [68.2, 15.2, 3.2],
        'Prefers gpt': [20.2, 81.2, 96.8]
    }
    
    q3_data = {
        'Prefers human': [18.4, 4.7, 0.0],
        'Neutral': [62.5, 5.8, 0.4],
        'Prefers gpt': [19.1, 89.5, 99.6]
    }
    
    # Spacing parameters
    bar_width = 0.2
    group_spacing = 0.8
    
    # Categories and positions with adjusted spacing
    categories = ['Prefers human', 'Neutral', 'Prefers gpt']
    x = np.arange(len(categories)) * group_spacing
    
    # Colorblind-friendly colors
    colors = ['#E15759', '#76B7B2', '#4E79A7']  # Red, Light blue, Darker blue
    
    def plot_question(ax, data, title):
        # Plot bars for each model
        for i, model in enumerate(['Human', 'GPT', 'QWEN']):
            values = [data[cat][i] for cat in categories]
            ax.bar(x + (i-1)*bar_width, values, bar_width, label=model, color=colors[i])
        
        # Customize subplot
        ax.set_ylabel('Percentage')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)
        
        # Adjust x-axis limits to control group spacing
        ax.set_xlim(min(x) - bar_width*2, max(x) + bar_width*2)
    
    # Create each subplot
    plot_question(ax1, q1_data, 'Q1 (Completeness)')
    plot_question(ax2, q2_data, 'Q2 (Correctness)')
    plot_question(ax3, q3_data, 'Q3 (Conciseness)')
    
    # Add legend to the figure
    fig.legend(['Human', 'GPT', 'QWEN'], 
              loc='upper center', 
              bbox_to_anchor=(0.5, 1.05),
              ncol=3)
    
    # Add overall title
    #fig.suptitle('Distribution of Preferences by Question\nAggregated Across Datasets', 
    #            y=1.15)
    
    # Adjust layout
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

    plot_preference_distribution()
    plt.savefig('llm_bias.png', dpi=300, bbox_inches='tight')