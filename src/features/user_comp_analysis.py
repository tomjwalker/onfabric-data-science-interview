"""
Module for Analyzing Large JSON Files Using OpenAI's GPT-4o-mini Model.

This module provides functionality to read a large JSON file incrementally and
analyze its contents using the OpenAI API. It processes the data in manageable
chunks due to the context length limitations of the model, combines the analyses,
and generates a comprehensive summary.

Functions:
- read_json_in_chunks: Reads a large JSON file in chunks.
- process_chunk: Analyzes a chunk of data using the OpenAI API.
- count_total_items: Counts the total number of items in the JSON file.
- main: Orchestrates the reading, processing, and summarizing of the data.

Dependencies:
- openai
- ijson
- json
- time
- typing
- tiktoken
- tqdm
"""
import openai
import ijson
import json
import time
import os
from typing import Iterator, List
import tiktoken
from tqdm import tqdm
from pathlib import Path
import yaml

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')  # Replace with your actual key or set the environment variable

class PromptManager:
    def __init__(self, config_path: str = "src/config/prompts.yaml"):
        with open(config_path, 'r') as f:
            self.prompts = yaml.safe_load(f)

    def get_chunk_prompt(self, data: str) -> str:
        template = self.prompts['prompts']['chunk_analysis']['template']
        return template.format(data=data)

    def get_summary_prompt(self, analyses: str) -> str:
        template = self.prompts['prompts']['final_summary']['template']
        return template.format(previous_analyses=analyses)

def read_json_in_chunks(file_path: str, chunk_size: int) -> Iterator[List[dict]]:
    """
    Reads a JSON file incrementally and yields chunks of data.

    Args:
        file_path (str): The path to the JSON file.
        chunk_size (int): The number of items to include in each chunk.

    Yields:
        Iterator[List[dict]]: An iterator over lists of JSON objects.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Adjust 'item' according to the structure of your JSON file
        objects = ijson.items(f, 'item')
        chunk: List[dict] = []
        for obj in objects:
            chunk.append(obj)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def count_total_items(file_path: str) -> int:
    """
    Counts the total number of items in the JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        int: The total number of items.
    """
    total_items = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'item')
        for _ in objects:
            total_items += 1
    return total_items

def process_chunk(chunk: List[dict]) -> str:
    """
    Sends a chunk of data to the OpenAI API for analysis and returns the result.

    Args:
        chunk (List[dict]): A list of JSON objects to analyze.

    Returns:
        str: The analysis result from the OpenAI model.
    """
    chunk_json = json.dumps(chunk)
    prompt = f"Analyze the following data and provide a summary:\n{chunk_json}"

    # Estimate the number of tokens in the prompt
    encoding = tiktoken.encoding_for_model('gpt-4')
    prompt_tokens = len(encoding.encode(prompt))

    # Set the maximum number of tokens for the response
    max_response_tokens = 500  # Adjust based on expected response length

    # Total tokens should not exceed the model's context length
    max_context_length = 8000  # For GPT-4o-mini
    if prompt_tokens + max_response_tokens > max_context_length:
        # Calculate how many tokens need to be removed
        tokens_to_remove = prompt_tokens + max_response_tokens - max_context_length
        # Estimate how many items to remove from the chunk
        avg_tokens_per_item = prompt_tokens / len(chunk)
        items_to_remove = int(tokens_to_remove / avg_tokens_per_item) + 1
        # Reduce the chunk size accordingly
        print(f"Reducing chunk size by {items_to_remove} items to fit within context length.")
        return process_chunk(chunk[:-items_to_remove])

    try:
        client = openai.OpenAI()  # Create client instance
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_response_tokens
        )
        analysis: str = response.choices[0].message.content
        return analysis
    except openai.RateLimitError:  # Updated exception class
        print("Rate limit exceeded. Waiting for 60 seconds before retrying...")
        time.sleep(60)
        return process_chunk(chunk)
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def main(verbose: bool = False, test_mode: bool = False, output_file: str = './data/processed/analysis_summary.txt') -> None:
    """
    Main function to read the JSON file, process each chunk, and generate a final summary.

    Args:
        verbose (bool): If True, prints progress updates.
        test_mode (bool): If True, only processes first 3 chunks.
        output_file (str): Path to save the final analysis.
    """
    file_path: str = './data/raw/search_history.json'
    chunk_size: int = 100

    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Optionally, get total items and chunks
    total_items = None
    total_chunks = None
    if verbose:
        total_items = count_total_items(file_path)
        total_chunks = (total_items + chunk_size - 1) // chunk_size
        print(f"Total items: {total_items}")
        print(f"Total chunks: {total_chunks}")

    chunk_summaries: List[str] = []
    chunks_processed = 0

    # Use tqdm progress bar if verbose
    if verbose and total_chunks:
        chunk_iterator = tqdm(
            read_json_in_chunks(file_path, chunk_size),
            total=total_chunks,
            desc="Processing chunks"
        )
    else:
        chunk_iterator = read_json_in_chunks(file_path, chunk_size)

    for chunk in chunk_iterator:
        analysis: str = process_chunk(chunk)
        if analysis:
            chunk_summaries.append(analysis)
            
        chunks_processed += 1
        if test_mode and chunks_processed >= 3:
            if verbose:
                print("Test mode: Stopping after 3 chunks")
            break

    combined_summary: str = '\n'.join(chunk_summaries)

    final_prompt: str = (
        f"Based on the following analyses, provide a comprehensive summary:\n{combined_summary}"
    )

    # Estimate tokens for final prompt
    encoding = tiktoken.encoding_for_model('gpt-4')
    final_prompt_tokens = len(encoding.encode(final_prompt))
    max_response_tokens = 1000  # Adjust as needed

    if final_prompt_tokens + max_response_tokens > 8000:
        print("Final combined summary is too long for the model's context length.")
        # Handle accordingly (e.g., summarize chunk summaries)

    try:
        if verbose:
            print("Generating final analysis...")
        client = openai.OpenAI()  # Create client instance
        final_response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=max_response_tokens
        )
        final_analysis: str = final_response.choices[0].message.content
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_analysis)
            
        if verbose:
            print(f"Final Analysis saved to: {output_file}")
            print("\nFirst few lines of analysis:")
            print("\n".join(final_analysis.split('\n')[:5]))
    except Exception as e:
        print(f"An error occurred while generating the final summary: {e}")

if __name__ == '__main__':
    # # Test mode with 10 chunks
    # main(verbose=True, test_mode=True)

    # # Full analysis
    # main(verbose=True, test_mode=False)

    # Custom output location
    main(verbose=True, test_mode=True, output_file='./data/processed/user_comp_analysis.txt')
