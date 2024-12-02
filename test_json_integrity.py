import json
import os

# Define the directory containing the generated JSON files
json_results_dir = "data/"
llm_names = ["gpt", "qwen", "mistral"]
questions = ["q1", "q2", "q3"]

# Function to load a JSON file and extract relevant data
def load_json_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Function to test if the number of tasks is the same across all JSON files
def test_task_count(json_results_dir, llm_names, questions):
    print("###### Task Count Consistency Test:\n")
    # Dictionary to store the task count for each question
    task_count = {question: [] for question in questions}

    # Iterate through all questions and LLM models
    for question in questions:
        for llm in llm_names:
            filename = os.path.join(json_results_dir, f"{question}_experiment_data_{llm}.json")
            json_data = load_json_data(filename)

            # Get the number of tasks (length of the "tasks" list)
            num_tasks = len(json_data['tasks'])
            task_count[question].append(num_tasks)

        # Check if all task counts for a given question are the same
        task_counts = task_count[question]
        if len(set(task_counts)) == 1:
            print(f"All LLMs have the same number of tasks for {question}: {task_counts[0]} tasks\n")
        else:
            print(f"Warning: Mismatched task counts for {question}: {task_counts}\n")

# Function to test context consistency across LLMs for each question
def test_context_consistency(json_results_dir, llm_names, questions):
    print("###### Context Consistency Test:\n")
    
    # Iterate through all questions and tasks
    for question in questions:
        context_mismatch_count = 0
        aligned_count = 0

        # Iterate through each task
        for task_id in range(262):  # Assuming there are 262 tasks
            task_contexts = []

            # Collect the context for each LLM
            for llm in llm_names:
                filename = os.path.join(json_results_dir, f"{question}_experiment_data_{llm}.json")
                json_data = load_json_data(filename)

                # Extract the context for the current task
                task_context = json_data['tasks'][task_id]['context']
                task_contexts.append(task_context)

            # Check if the context is the same across all LLMs
            if len(set(task_contexts)) == 1:
                aligned_count += 1
            else:
                context_mismatch_count += 1

        # Print summary for context consistency for each question
        print(f"Summary for {question} - Context Consistency:")
        print(f"Aligned contexts: {aligned_count} / 262")
        print(f"Mismatched contexts: {context_mismatch_count} / 262")
        print("\n" + "-" * 50)

# Function to test alignment of first 5 responses across LLMs for each question
def test_human_response_alignment(json_results_dir, llm_names, questions):
    print("###### Human Response Alignment Test:\n")
    for question in questions:
        misaligned_count = 0
        aligned_count = 0

        # Iterate through each task
        for task_id in range(262):  # Assuming there are 262 tasks
            task_responses = []

            # Collect the first 5 responses for each LLM
            for llm in llm_names:
                filename = os.path.join(json_results_dir, f"{question}_experiment_data_{llm}.json")
                json_data = load_json_data(filename)
                
                # Extract the first 5 responses for the current task
                first_5_responses = json_data['tasks'][task_id]['responses'][:5]
                task_responses.append(first_5_responses)

            # Check if the first 5 responses are the same across all LLMs
            if len(set(tuple(r) for r in task_responses)) == 1:
                aligned_count += 1
            else:
                misaligned_count += 1

        # Print summary for response alignment for each question
        print(f"Summary for {question} - Response Alignment:")
        print(f"Aligned tasks: {aligned_count} / 262")
        print(f"Misaligned tasks: {misaligned_count} / 262")
        print("\n" + "-" * 50)

# Run the test for task count consistency
test_task_count(json_results_dir, llm_names, questions)

# Run the test for context consistency
test_context_consistency(json_results_dir, llm_names, questions)

# Run the test for human response alignment
test_human_response_alignment(json_results_dir, llm_names, questions)