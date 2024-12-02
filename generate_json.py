import os
import pandas as pd
import json

def load_and_combine_data(raw_data_dir, llm_names):
    """
    Loads and combines multiple CSV files for each LLM from the specified directory.

    Args:
    - raw_data_dir (str): The directory where raw CSV files are located.
    - llm_names (list): List of LLM names whose CSV files need to be read.

    Returns:
    - pandas.DataFrame: Combined DataFrame containing all data from the CSV files.
    """
    csv_files = [os.path.join(raw_data_dir, f"{llm}_judge.csv") for llm in llm_names]
    raw_data_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(raw_data_list, ignore_index=True).drop_duplicates()
    return combined_df

def data_prep(df, llm_names):
    """
    Filters the data to ensure each item is read by exactly 5 humans and a specified number of LLMs.
    Human readers are identified by reader names starting with a capital letter (e.g. "R_1KuQeqzaAixl0UF")
    while LLM readers are identified by names starting with a lowercase letter (e.g. "gpt").

    Args:
    - df (pandas.DataFrame): Raw input DataFrame containing columns: `dataset`, `inputs`, `output`, `target`, and `reader`.
    - llm_names (list): List of LLM names.

    Returns:
    - pandas.DataFrame: A DataFrame filtered to include only items read by exactly 5 human readers and len(llm_names) LLMs.
    """
    # Create unique item_id for easy reference
    df['item_id'] = df.groupby(['dataset', 'inputs', 'output', 'target']).ngroup()

    # Ensure each item is read by exactly 5 humans and all LLMs
    df = df.groupby('item_id').filter(
        lambda group: (
            group['reader'].str[0].str.isupper().sum() == 5 and  # 5 humans
            group['reader'].str[0].str.islower().sum() == len(llm_names)
        )
    )
    return df

def generate_json_for_all(df, llm_names, json_results_dir):
    """
    Generates JSON files for each question and LLM. Each JSON file contains 5 human responses and 1 LLM response per task.

    Args:
    - df (pandas.DataFrame): DataFrame containing data to process.
    - llm_names (list): List of LLM names (e.g., ["gpt", "qwen", "mistral"]).
    - json_results_dir (str): Directory to save the generated JSON files.

    Returns:
    - None: Saves 9 JSON files (one for each question and LLM) in the specified directory.
    """
    criterion_description = {
        "q1": "Which summary more completely captures important information? This compares the summaries' recall, i.e., the amount of clinically significant detail retained from the input text.",
        "q2": "Which summary includes less false information? This compares the summaries' precision, i.e., instances of fabricated information.",
        "q3": "Which summary contains less non-important information? This compares which summary is more condensed, as the value of a summary decreases with superfluous information."
    }

    # Filter rows for humans
    df_humans = df[df['reader'].str[0].str.isupper()]

    for llm in llm_names:
        # Filter rows for current LLM
        df_llm = df[df['reader'] == llm]
        df_subset = pd.concat([df_humans, df_llm])

        for question in ["q1", "q2", "q3"]:
            summarized_data = df_subset.groupby('item_id').agg(
                context = ('inputs', lambda x: f"Input Text:\n{x.iloc[0]}\nTarget Summary:\n{x.iloc[1]}\nOutput Summary:\n{x.iloc[2]}\nCriterion: {criterion_description[question]}\nPlease answer with '1' if you prefer the Target Summary, '-1' if you prefer the Output Summary, or '0' if you have no preference."),
                responses = (question, list)
            ).reset_index()

            json_structure = {
                "task_description": f"You are an expert summarization evaluator for clinical data. \nCriterion used: {question}\nLLM used: {llm}",
                "agent_perspectives": [{"reading": None, "strategy": None} for _ in range(6)],
                "tasks": [
                    {"context": row['context'], "responses": row['responses']} 
                    for _, row in summarized_data.iterrows()
                ]
            }

            # Write JSON structure to a file
            json_filename = os.path.join(json_results_dir, f"{question}_experiment_data_{llm}.json")
            with open(json_filename, 'w') as json_file:
                json.dump(json_structure, json_file, indent=4)

            print(f"Saved: {json_filename}")

# Main function to execute the process
def main(raw_data_dir, llm_names, json_results_dir):
    df_raw_all = load_and_combine_data(raw_data_dir, llm_names)
    df_filtered = data_prep(df_raw_all, llm_names)
    generate_json_for_all(df_filtered, llm_names, json_results_dir)

# Call the main function
raw_data_dir = "data_raw/"
json_results_dir = "data/"
llm_names = ["gpt", "qwen", "mistral"]
main(raw_data_dir, llm_names, json_results_dir)