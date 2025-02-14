import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv('.env')

def gpt_answers():

    """
    Extracts the first 500 answerable questions from the SQuAD2.0 Dev Set v2.0
    and uses the OpenAI API to run GPT-4o-Mini in batch mode to generate answers to all the
    answerable questions and save the responses to a json file.
    """

    # Load SQuAD2.0 dataset
    with open('dev-v2.0.json') as f:
        data = json.load(f)

    questions = []
    # Extract the first 500 answerable questions
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if not qa["is_impossible"]:  # Skip unanswerable questions
                    question = qa["question"]
                    questions.append(qa["question"])
                if len(questions) >= 500:
                    break
            if len(questions) >= 500:
                break
        if len(questions) >= 500:
            break

    print(f"Extracted {len(questions)} questions.")

    # System prompt
    system_prompt = "You are an intelligent AI. Answer all questions concisely and to the best of your ability."
    user_prompt = "Answer this question concisely and to the best of your ability: {question}"

    # Create batch request file
    tasks = []
    for id_num, question in enumerate(questions):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(question=question)}
        ]

        task = {
            "custom_id": f"question_{id_num + 1}",  # Ensures uniqueness
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "messages": messages
            }
        }
        tasks.append(task)

    # Writing a local json file to store the tasks
    with open("gpt-answers-input-batch.jsonl", 'w') as jfile:
        for task in tasks:
            jfile.write(json.dumps(task) + '\n')

    # Establish OpenAI Client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # upload our batch file to OpenAI
    batch_file = client.files.create(
        file=open("gpt-answers-input-batch.jsonl", 'rb'),
        purpose='batch'
    )

    # Run the batch using the completions endpoint
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    # loop until the status of our batch is completed
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job.id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        time.sleep(1)

    print("Done processing batch.")
    print(batch_job)
    print("Writing data...")
    print(check)

    # Write the results to a local file, again, jsonl format
    result = client.files.content(check.output_file_id).content
    output_file_name = "gpt-answers-output-batch.jsonl"
    with open(output_file_name, 'wb') as file:
        file.write(result)

    # load the output file, extract each sample output, and append to a list
    results = []
    with open(output_file_name, 'r') as file:
        for line in file:
            # this converts the string into a Json object
            json_object = json.loads(line.strip())
            results.append(json_object)

    # Show the responses
    for item in results:
        print("Model's Response:")
        print('\t', item['response']['body']['choices'][0]['message']['content'])

# Run the function
gpt_answers()