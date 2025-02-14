import os
import json

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

def llama_answers():

    """
    Extracts the first 500 answerable questions from the SQuAD2.0 Dev Set v2.0
    and uses Azure to run Llama 3.2 11B Vision Instruct to serially generate
    answers to all the answerable questions and save the responses to a json file.
    """

    # Establish Client
    client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_MLSTUDIO_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_MLSTUDIO_KEY"]),
    )

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

    # Open the output file in append mode
    with open('llama-answers.json', 'a') as output_file:
        for i, question in enumerate(questions, 1):
            # Submit the question to the Llama model
            response = client.complete(
                messages=[
                    SystemMessage(
                        content=(
                            "You are an intelligent AI. Answer all questions concisely and to the best of your ability."
                        )
                    ),
                    UserMessage(content=question),
                ],
            )
            # Structure the result
            result = {
                "question": question,
                "response": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
            print(f"{i} / {len(questions)} questions answered: {question}")
            output_file.write(json.dumps(result) + '\n')

    # print the results
    print("Model's Response:")
    print('\t', response.choices[0].message.content)
    print()
    print(f"Input Tokens:  {response.usage.prompt_tokens}")
    print(f"Output Tokens: {response.usage.completion_tokens}")
    print(f"Cost: ${response.usage.prompt_tokens * 0.0003 / 1000 + response.usage.completion_tokens * 0.00061 / 1000}")

llama_answers()