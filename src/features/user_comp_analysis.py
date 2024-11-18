"""
User Behaviour and Fashion Analysis Module

A comprehensive analysis tool that processes large JSON datasets (e.g., user browsing history)
to extract fashion insights and user behaviour patterns using OpenAI's GPT models.

Key Features:
- Chunked processing of large JSON files with memory efficiency
- Fashion entity extraction (brands, products, styles)
- Topic analysis and trend identification
- Vector database caching using ChromaDB
- Rate limiting and cost estimation
- Comprehensive summary generation

Processing Steps:
1. Reads JSON data in manageable chunks
2. For each chunk:
   - Performs general content analysis
   - Extracts fashion-specific entities and topics
   - Caches results in vector database
3. Combines analyses to generate:
   - Fashion entity frequencies
   - Topic distribution
   - Comprehensive summary
4. Saves results to structured output files

Command Line Usage:
    # Full analysis
    $ python user_comp_analysis.py --input data/raw/search_history.json --verbose

    # Test run (2 chunks only)
    $ python user_comp_analysis.py --test --verbose

    # Custom output location
    $ python user_comp_analysis.py -i input.json -o output/analysis.txt

Arguments:
    --input, -i    : Input JSON file path
    --output, -o   : Output file path (default: ./data/processed/analysis_summary.txt)
    --verbose, -v  : Enable detailed progress logging
    --test, -t     : Run in test mode (process only 2 chunks)

Output Files:
    - analysis_summary.txt: Overall analysis and insights
    - fashion_analysis.json: Structured fashion entity and topic data

Dependencies:
    - openai: LLM integration
    - ijson: Memory-efficient JSON processing
    - chromadb: Vector database for caching
    - tiktoken: Token counting for cost estimation
    - tqdm: Progress tracking
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

class ModelConfig:
    """Central configuration for OpenAI model settings"""
    
    DEFAULT_MODEL = "gpt-4o-mini"
    
    # Model context windows
    MAX_TOKENS = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-3.5-turbo": 4096,
        "gpt-4-turbo": 128000,
        "gpt-4o-mini": 8000
    }
    
    # Cost per 1k tokens (input, output) in USD
    COSTS_PER_1K_TOKENS = {
        "gpt-4": (0.03, 0.06),
        "gpt-4-32k": (0.06, 0.12),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "gpt-4o-mini": (0.000150, 0.000600)
    }
    
    @classmethod
    def get_context_length(cls, model: str) -> int:
        """Get the maximum context length for a model"""
        return cls.MAX_TOKENS.get(model, cls.MAX_TOKENS[cls.DEFAULT_MODEL])
    
    @classmethod
    def get_cost_rates(cls, model: str) -> tuple[float, float]:
        """Get the cost rates (input, output) for a model"""
        return cls.COSTS_PER_1K_TOKENS.get(model, cls.COSTS_PER_1K_TOKENS[cls.DEFAULT_MODEL])

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
    
    # Add exponential backoff for rate limiting
    max_retries = 5
    base_delay = 1  # Start with 1 second delay
    
    for attempt in range(max_retries):
        try:
            # Check if similar analysis exists
            collection = chroma_client.get_or_create_collection("chunk_analyses")
            similar_results = collection.query(
                query_texts=[chunk_json],
                n_results=1,
                where={"similarity": {"$gt": 0.95}}
            )
            
            if similar_results and similar_results['documents']:
                # Even if we found a cached analysis, still update fashion entities and topics
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
            
        except openai.RateLimitError:
            if attempt == max_retries - 1:  # Last attempt
                raise
            delay = (2 ** attempt) * base_delay  # Exponential backoff
            print(f"Rate limit exceeded. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}")
            time.sleep(delay)
        except Exception as e:
            print(f"Error in summary generation: {str(e)}")
            return ""

def process_with_openai(chunk: List[dict], prompt_manager: PromptManager, model: str = ModelConfig.DEFAULT_MODEL) -> str:
    """Sends a chunk of data to the OpenAI API for analysis"""
    chunk_json = json.dumps(chunk)
    prompt = prompt_manager.get_chunk_prompt(chunk_json)

    encoding = tiktoken.encoding_for_model(model)
    prompt_tokens = len(encoding.encode(prompt))
    max_response_tokens = 500
    max_context_length = ModelConfig.get_context_length(model)

    if prompt_tokens + max_response_tokens > max_context_length:
        tokens_to_remove = prompt_tokens + max_response_tokens - max_context_length
        avg_tokens_per_item = prompt_tokens / len(chunk)
        items_to_remove = int(tokens_to_remove / avg_tokens_per_item) + 1
        print(f"Reducing chunk size by {items_to_remove} items to fit within context length.")
        return process_with_openai(chunk[:-items_to_remove], prompt_manager, model)

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_response_tokens
        )
        return response.choices[0].message.content
    except openai.RateLimitError:
        print("Rate limit exceeded. Waiting for 60 seconds before retrying...")
        time.sleep(60)
        return process_with_openai(chunk, prompt_manager, model)
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

def estimate_tokens_and_cost(chunk: List[dict], model: str = ModelConfig.DEFAULT_MODEL) -> tuple[int, float]:
    """Estimate tokens and cost for processing a chunk."""
    encoding = tiktoken.encoding_for_model(model)
    chunk_str = json.dumps(chunk)
    token_count = len(encoding.encode(chunk_str))
    
    input_cost, output_cost = ModelConfig.get_cost_rates(model)
    estimated_output_tokens = 500  # Approximate response length
    
    total_cost = (token_count * input_cost / 1000) + (estimated_output_tokens * output_cost / 1000)
    return token_count + estimated_output_tokens, total_cost

def main(verbose: bool = False, test_mode: bool = False, output_file: str = './data/processed/analysis_summary.txt', input_file: str = None, model: str = ModelConfig.DEFAULT_MODEL) -> None:
    """
    Main function to read the JSON file, process each chunk, and generate a final summary.
    
    Args:
        verbose (bool): Enable verbose logging
        test_mode (bool): Run in test mode (process only 2 chunks)
        output_file (str): Path to output file
        input_file (str): Path to input JSON file (optional)
    """
    # Ensure all paths are Path objects and absolute
    base_dir = Path(__file__).parent.parent.parent  # Gets the project root directory
    file_path = Path(input_file) if input_file else base_dir / 'data/raw/search_history.json'
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
        chroma_client=chroma_client,
        model=model
    )

    if verbose:
        print("Counting total items...")
        total_items = count_total_items(str(file_path))
        estimated_chunks = total_items // 100  # Assuming chunk_size of 100
        print(f"Found {total_items} items ({estimated_chunks} chunks)")
        
        # Estimate total cost
        with open(str(file_path), 'r', encoding='utf-8') as f:
            sample_chunk = []
            for i, line in enumerate(ijson.items(f, 'item')):
                sample_chunk.append(line)
                if i >= 99:  # Get a sample chunk of 100 items
                    break
        
        tokens, cost_per_chunk = estimate_tokens_and_cost(sample_chunk, model=model)
        total_estimated_cost = cost_per_chunk * estimated_chunks * 3  # *3 for entity, topic, and summary analysis
        print(f"Estimated total cost: ${total_estimated_cost:.2f}")
        print(f"Estimated tokens per chunk: {tokens}")
        
        proceed = input("Do you want to proceed? (y/n): ")
        if proceed.lower() != 'y':
            print("Aborting...")
            return

    chunk_summaries: List[str] = []
    chunks_processed = 0
    
    # Create progress bar if verbose
    progress_bar = tqdm(total=estimated_chunks) if verbose else None

    for chunk in read_json_in_chunks(str(file_path), chunk_size=100):
        if verbose:
            print(f"\nProcessing chunk {chunks_processed + 1}/{estimated_chunks}")
            
        # Process chunk with both general and fashion analysis
        analysis_results = process_chunk(chunk, prompt_manager, chroma_client, fashion_analyzer)
        
        if analysis_results:
            chunk_summaries.append(analysis_results)
            
        chunks_processed += 1
        if progress_bar:
            progress_bar.update(1)
            
        if test_mode and chunks_processed >= 2:
            if verbose:
                print("\nTest mode: Stopping after 2 chunks")
            break

    if progress_bar:
        progress_bar.close()

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

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze JSON data using GPT-4o-mini')
    parser.add_argument('--input', '-i', help='Path to input JSON file')
    parser.add_argument('--output', '-o', default='./data/processed/analysis_summary.txt', help='Path to output file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--test', '-t', action='store_true', help='Run in test mode (process only 2 chunks)')
    
    args = parser.parse_args()
    
    main(
        verbose=args.verbose,
        test_mode=args.test,
        output_file=args.output,
        input_file=args.input
    )

# if __name__ == '__main__':
#     """Test the complete pipeline with a small subset of data"""
#     print("Testing pipeline with 2 chunks...")
#     main(verbose=True, test_mode=True)

if __name__ == '__main__':
    """Test the complete pipeline with a small subset of data"""
    print("Testing pipeline with 2 chunks...")
    main(verbose=True, test_mode=True, output_file='./data/processed/test_analysis.txt')
