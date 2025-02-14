import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

# Load Env
load_dotenv('.env')

def gpt_grading():
    """
    Uses the OpenAI API to run GPT-4o in batch mode in order to
    score the responses of GPT-4o-mini as either true or false and
    save the outputs in a json file providing reasoning.
    """

    # Retrieve the OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")

    with open('dev-v2.0.json') as f:
        correct_data = json.load(f)

    # Load gpt's responses from the output file
    with open("gpt-answers-output-batch.jsonl") as f:
        gpt_data = [json.loads(line) for line in f if line.strip()]

    gpt_answers = []  # Define list before loop
    for data in gpt_data:
        # Extract the question from custom_id
        question = data["custom_id"]  # Use custom_id as identifier

        # Extract the responses
        response_content = data["response"]["body"]["choices"][0]["message"]["content"]

        # Append to gpt_answers list
        gpt_answers.append({
            "question": question,
            "response": response_content
        })

    correct_answers = [] # Define list before loop

    for article in correct_data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if not qa['is_impossible'] and qa['answers']:
                    answer = qa['answers'][0]['text'].lower()
                    correct_answers.append(answer)
                if len(correct_answers) >= 500:
                    break
            if len(correct_answers) >= 500:
                break
        if len(correct_answers) >= 500:
            break

    # Define the JSON schema for the structured output.
    json_schema = {
        "name": "grading_output",
        "schema": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "A short explanation of why the student's answer was correct or incorrect."
                },
                "score": {
                    "type": "boolean",
                    "description": "True if the student's answer is correct, false otherwise."
                }
            },
            "required": ["explanation", "score"],
            "additionalProperties": False
        },
        "strict": True
    }

    # Define prompts
    system_prompt = """You are a teacher tasked with determining whether a student’s answer to a question was correct, based on a set of possible correct answers. You must only use the provided possible correct answers to determine if the student’s response was correct."""
    user_prompt = """Question: {question}
    Student's Response: {student_response}
    Possible Correct Answers: {correct_answer}

    Your response should be a valid JSON in the following format:
    {{
    "explanation": "(str): A short explanation of why the student’s answer was correct or incorrect.",
    "score": "(bool): true if the student’s answer was correct, false if it was incorrect."
    }}"""

    tasks = []  # Create grading tasks for each question and response pair
    for idx, (correct_answer, gpt_answer) in enumerate(zip(correct_answers, gpt_answers), 1):
        if correct_answer is None:
            continue
        question = gpt_answer['question']
        student_response = gpt_answer['response']

        # Format the user prompt with the question, student response, and correct answer
        formatted_user_prompt = user_prompt.format(
            question=question,
            student_response=student_response,
            correct_answer=correct_answer
        )

        # Prepare the messages to be sent to the model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user_prompt}
        ]

        # Create the grading task
        custom_id = f"{idx}. {question}"
        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "temperature": 0.2,
                "messages": messages,
            }
        }
        tasks.append(task)

    # Write the batch tasks to a JSONL file.
    input_filename = "gpt-grading-input-batch.jsonl"
    with open(input_filename, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    # Initialize the OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Upload the batch file to OpenAI
    batch_file = client.files.create(
        file=open(input_filename, 'rb'),
        purpose='batch'
    )

    # Create a batch job for grading
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    # Poll the batch job status until completion
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job.id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        time.sleep(1)

    print("Batch processing completed.")

    # Retrieve and save the grading results
    result = client.files.content(check.output_file_id).content
    output_file_name = "gpt-4o-mini-2-13-2025-hw3.jsonl"
    with open(output_file_name, 'wb') as file:
        file.write(result)

    print(f"Grading results saved to {output_file_name}.")


gpt_grading()