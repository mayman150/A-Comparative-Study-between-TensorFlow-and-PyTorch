from openai import OpenAI
from decouple import config
import pandas as pd
import argparse
import re
import inspect


import datetime
import json
import os
import sys

from csv import DictReader
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

regex = re.compile(r"\d+\.\s")

def setup():
    client = OpenAI(
        api_key=config("SECRET_KEY")
    )
    
    with open(config("PROMPT_LOC"), 'r') as f:
        initial_prompt = f.read()
    return client, initial_prompt

@retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(10))
def run_chatgpt(model, temp, meta_prompt, max_tokens):
    """
    Run the ChatGPT model on the input prompt.
    """
    # Define the parameters for the text generation
    completions = OpenAI.Completion.create(
        engine=model,
        prompt=meta_prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temp,
    )
    gen_prompt = completions.choices[0].text.strip().lower()
    # Print the generated text
    print("The meta prompt is --> ", meta_prompt)
    print("LLM output is --> ", gen_prompt)
    return gen_prompt


def generate_issue_prompts(csv_file: str, no_of_samples: int):
    data = pd.read_csv(csv_file)

    samples = data.sample(no_of_samples)
    samples.columns = map(str.lower, samples.columns)

    prompts = list()
    for _, row in samples.iterrows():
        tags = str(row.get("tags"))
        title = str(row.get("issue title"))
        body = inspect.cleandoc(str(row.get("issue body")))
        prompt = f'Q:\nIssue Title: """{title}"""\nIssue Tags: """{tags}"""\nIssue Body:\n"""\n{body}\n"""'

        prompts.append(re.escape(inspect.cleandoc(prompt)))

    return prompts

def main():
    parser = argparse.ArgumentParser(allow_abbrev=True)
    client, initial_prompt = setup()
    parser.add_argument("csv_files", nargs="+", help="Locations of the csv files of issues.")
    parser.add_argument("-n", "--no_of_samples", type=int, default=100, metavar="", help="Number of samples from each csv file. Default to 100")
    args = parser.parse_args()


    issue_prompts = []
    for csv_file in args.csv_files:
        print("Generating prompt for file: " + csv_file)
        issue_prompts += generate_issue_prompts(csv_file, args.no_of_samples)
    gpt_name = "gpt-3.5-turbo-1106"
    for issue_prompt in tqdm(issue_prompts):
        full_prompt = initial_prompt + ' ' +issue_prompt
        run_chatgpt(
            model=gpt_name,
            temp=0,
            meta_prompt=full_prompt,
            max_tokens=16000,
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        exit(0)
    except FileNotFoundError as e:
        print("Unable to locate file: ", str(e.filename))
        print("Check the file specified in argument, and .env file")
        exit(e.errno)