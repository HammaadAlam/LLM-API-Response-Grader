import json

def calculate_average_correctness(file_path):
    """
    Reads each line of the given file and extracts the score
    (true or false) in order to determine the accuracy of the
    models
    :param file_path: Path to the json file
    :return: A float value of the accuracy of the models
    """
    # Initialize variables to keep track of the number of correct answers
    correct_count = 0
    total_count = 0

    # Open output JSONL file
    with open(file_path, 'r') as f:
        for line in f:
            # Load each line as a JSON object
            data = json.loads(line)

            # Extract the correctness score from the response (true or false)
            response_content = data["response"]["body"]["choices"][0]["message"]["content"]

            # Parse the content
            response_json = json.loads(response_content.strip("```json\n").strip("```"))

            # Check if the score is True or False and increment count accordingly
            if response_json.get("score") is True:
                correct_count += 1
            total_count += 1

    # Calculate the average correctness (percentage of correct answers)
    if total_count > 0:
        average_correctness = (correct_count / total_count) * 100
        print(f"Acuracy for {file_path}: {average_correctness:.2f}%")
    else:
        print("No tasks to evaluate.")


# Call the function to determine accuracy
file_path = "gpt-4o-mini-2-13-2025-hw3.jsonl"
calculate_average_correctness(file_path)

file_path = "llama-2-13-2025-hw3.jsonl"
calculate_average_correctness(file_path)