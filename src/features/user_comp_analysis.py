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
from typing import Iterator, List, Dict, Any
import tiktoken
from tqdm import tqdm
from pathlib import Path
import yaml
import chromadb
from chromadb.utils import embedding_functions
import hashlib
from src.features.fashion_analysis import FashionAnalyzer

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')  # Replace with your actual key or set the environment variable

class PromptManager:
    def __init__(self, config_path: str = "src/config/prompts.yaml"):
        with open(config_path, 'r') as f:
            self.prompts = yaml.safe_load(f)

    def get_chunk_prompt(self, data: str) -> str:
        template = self.prompts['prompts']['chunk_analysis']['template']
        return template.format(data=data)

    def get_fashion_entity_prompt(self, data: str, current_entities: dict) -> str:
        template = self.prompts['prompts']['fashion_entity_extraction']['template']
        return template.format(
            data=data,
            current_entities=json.dumps(current_entities, indent=2)
        )

    def get_topic_extraction_prompt(self, data: str, current_topics: list) -> str:
        template = self.prompts['prompts']['topic_extraction']['template']
        return template.format(
            data=data,
            current_topics=json.dumps(current_topics, indent=2)
        )

    def get_summary_prompt(self, analyses: str) -> str:
        template = self.prompts['prompts']['final_summary']['template']
        return template.format(
            previous_analyses=analyses,
            fashion_entities=json.dumps(self.fashion_entities) if hasattr(self, 'fashion_entities') else "No fashion entities found",
            topic_frequencies=json.dumps(self.topic_frequencies) if hasattr(self, 'topic_frequencies') else "No topic frequencies found"
        )

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

def process_chunk(chunk: List[dict], prompt_manager: PromptManager, chroma_client, fashion_analyzer: FashionAnalyzer) -> Dict[str, Any]:
    """
    Process chunk and store in vector DB
    """
    chunk_json = json.dumps(chunk)
    chunk_hash = hashlib.md5(chunk_json.encode()).hexdigest()
    
    # Check if similar analysis exists
    collection = chroma_client.get_or_create_collection("chunk_analyses")
    similar_results = collection.query(
        query_texts=[chunk_json],
        n_results=1,
        where={"similarity": {"$gt": 0.95}}
    )
    
    if similar_results and similar_results['documents']:
        # Even if we found a cached analysis, still update fashion entities and topics
        # as we want to track frequencies
        fashion_analyzer.analyze_chunk(chunk)
        return similar_results['documents'][0]
    
    # If no similar analysis, process with OpenAI and fashion analyzer
    analysis_results = fashion_analyzer.analyze_chunk(chunk)
    
    # Store new analysis with enhanced metadata
    collection.add(
        documents=[analysis_results['summary']],
        metadatas=[{
            "chunk_hash": chunk_hash,
            "timestamp": time.time(),
            "entities": json.dumps(analysis_results['entities']),
            "topics": json.dumps(analysis_results['topics'])
        }],
        ids=[chunk_hash]
    )
    
    return analysis_results['summary']

def process_with_openai(chunk: List[dict], prompt_manager: PromptManager) -> str:
    """
    Sends a chunk of data to the OpenAI API for analysis and returns the result.

    Args:
        chunk (List[dict]): A list of JSON objects to analyze.
        prompt_manager (PromptManager): Manager for handling prompt templates.

    Returns:
        str: The analysis result from the OpenAI model.
    """
    chunk_json = json.dumps(chunk)
    prompt = prompt_manager.get_chunk_prompt(chunk_json)

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
        return process_with_openai(chunk[:-items_to_remove], prompt_manager)

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
        return process_with_openai(chunk, prompt_manager)
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def generate_final_summary(chunk_summaries: List[str], fashion_analysis: Dict, prompt_manager: PromptManager) -> str:
    """
    Generate a final summary from all chunk analyses and fashion analysis data.
    
    Args:
        chunk_summaries (List[str]): List of summaries from each chunk
        fashion_analysis (Dict): Dictionary containing fashion analysis results
        prompt_manager (PromptManager): Manager for handling prompt templates
    
    Returns:
        str: Final comprehensive summary
    """
    # Combine all chunk summaries
    combined_analyses = "\n\n".join(chunk_summaries)
    
    # Add fashion analysis data to prompt manager temporarily
    prompt_manager.fashion_entities = fashion_analysis['raw_entities']
    prompt_manager.topic_frequencies = fashion_analysis['topic_frequencies']
    
    # Get the summary prompt and process with OpenAI
    prompt = prompt_manager.get_summary_prompt(combined_analyses)
    
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        final_summary = response.choices[0].message.content
        
        # Add fashion analysis statistics
        final_summary += "\n\nFashion Analysis Statistics:\n"
        final_summary += f"Total unique entities: {sum(len(entities) for subcats in fashion_analysis['raw_entities'].values() for entities in subcats.values())}\n"
        final_summary += f"Total topics identified: {len(fashion_analysis['raw_topics'])}\n"
        
        return final_summary
        
    except Exception as e:
        print(f"Error generating final summary: {e}")
        return "Error generating final summary. Please check the individual chunk analyses and fashion analysis data."

def main(verbose: bool = False, test_mode: bool = False, output_file: str = './data/processed/analysis_summary.txt') -> None:
    """
    Main function to read the JSON file, process each chunk, and generate a final summary.
    """
    # Ensure all paths are Path objects and absolute
    base_dir = Path(__file__).parent.parent.parent  # Gets the project root directory
    file_path = base_dir / 'data/raw/search_history.json'
    output_path = base_dir / 'data/processed'
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Input file path: {file_path}")
        print(f"Output directory: {output_path}")
        print("Initializing analysis components...")
    
    # Initialize components
    prompt_manager = PromptManager()
    chroma_client = chromadb.Client()
    fashion_analyzer = FashionAnalyzer(
        prompt_manager=prompt_manager,
        chroma_client=chroma_client
    )

    chunk_summaries: List[str] = []
    chunks_processed = 0

    if verbose:
        print("Starting chunk processing...")
        
    for chunk in read_json_in_chunks(str(file_path), chunk_size=100):
        if verbose:
            print(f"\nProcessing chunk {chunks_processed + 1}")
            
        # Process chunk with both general and fashion analysis
        analysis_results = process_chunk(chunk, prompt_manager, chroma_client, fashion_analyzer)
        
        if analysis_results:
            chunk_summaries.append(analysis_results)
            
        chunks_processed += 1
        if test_mode and chunks_processed >= 2:
            if verbose:
                print("\nTest mode: Stopping after 2 chunks")
            break

    if verbose:
        print("\nGenerating final analysis...")

    # Save fashion analysis results
    fashion_analysis_output = {
        'entity_frequencies': fashion_analyzer.get_entity_frequencies(),
        'topic_frequencies': {topic: fashion_analyzer.fashion_topics.count(topic) 
                            for topic in set(fashion_analyzer.fashion_topics)},
        'raw_entities': fashion_analyzer.fashion_entities,
        'raw_topics': fashion_analyzer.fashion_topics
    }
    
    fashion_analysis_path = output_path / 'fashion_analysis.json'
    if verbose:
        print(f"Saving fashion analysis to: {fashion_analysis_path}")
    
    with open(fashion_analysis_path, 'w', encoding='utf-8') as f:
        json.dump(fashion_analysis_output, f, indent=2)

    # Save general analysis
    analysis_path = output_path / 'analysis_summary.txt'
    if verbose:
        print(f"Saving general analysis to: {analysis_path}")
    
    final_summary = generate_final_summary(chunk_summaries, fashion_analysis_output, prompt_manager)
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(final_summary)

    if verbose:
        print("\nProcessing complete. Files saved:")
        print(f"1. Fashion Analysis: {fashion_analysis_path}")
        print(f"2. General Analysis: {analysis_path}")
        
        print("\nSample of extracted entities:")
        for category, subcategories in fashion_analyzer.fashion_entities.items():
            print(f"\n{category.upper()}:")
            for subcategory, entities in subcategories.items():
                print(f"  {subcategory}: {list(set(entities))[:3]}")

# if __name__ == '__main__':
#     """Test the complete pipeline with a small subset of data"""
#     print("Testing pipeline with 2 chunks...")
#     main(verbose=True, test_mode=True)

if __name__ == '__main__':
    """Test the complete pipeline with a small subset of data"""
    print("Testing pipeline with 2 chunks...")
    main(verbose=True, test_mode=True, output_file='./data/processed/test_analysis.txt')
