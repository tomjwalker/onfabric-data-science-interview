"""Fashion catalog processing and search demonstration script.

This script showcases the usage of the catalog processing pipeline, which:
1. Processes fashion product data using OpenAI embeddings
2. Extracts structured entities (brand, category, price tier)
3. Stores processed items in a vector database (ChromaDB)
4. Enables semantic search with filtering

Usage:
    python process_catalog.py [--catalog-path PATH] [--verbose] [--test]

Arguments:
    --catalog-path    Path to JSON catalog file (default: data/processed/fashion_catalog_sampled.json)
    --verbose, -v     Enable detailed processing output
    --test, -t        Run in test mode

Input JSON format:
    [
        {
            "title": str,
            "description": str,
            "brand": str,
            "product_url": str,
            "gender": str,
            ...
        },
        ...
    ]

Environment Variables:
    OPENAI_API_KEY    Required for OpenAI API access
    CATALOG_PATH      Optional default catalog path

Example:
    python process_catalog.py --catalog-path data/my_catalog.json --verbose 
"""
import json
from pathlib import Path
import logging
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import argparse

from src.data_processing.catalog_processor import CatalogProcessor
from src.config.model_config import ModelConfig, VectorDBConfig
from src.data_processing.schema import CatalogItem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and set up OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_catalog(file_path: str) -> List[Dict]:
    """Load the fashion catalog from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Just return the raw dictionary data
            return data
    except Exception as e:
        logger.error(f"Error loading catalog file: {str(e)}")
        raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process fashion catalog data')
    parser.add_argument(
        '--catalog-path',
        type=str,
        default=os.getenv('CATALOG_PATH', 'data/processed/fashion_catalog_sampled.json'),
        help='Path to the fashion catalog JSON file'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--test',
        '-t',
        action='store_true',
        help='Run in test mode'
    )
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize configurations
    model_config = ModelConfig()
    db_config = VectorDBConfig()
    
    # Initialize the catalog processor
    processor = CatalogProcessor(model_config, db_config)
    
    # Load catalog data using the configured path
    catalog_path = Path(args.catalog_path)
    logger.info(f"Loading catalog from {catalog_path}")
    raw_catalog = load_catalog(catalog_path)
    
    # Get cost estimate before processing
    estimated_tokens, estimated_cost = processor.estimate_processing_cost(raw_catalog)
    logger.info("\nEstimated Processing Requirements:")
    logger.info(f"Total items: {len(raw_catalog)}")
    logger.info(f"Estimated tokens: {estimated_tokens:,}")
    logger.info(f"Estimated cost: ${estimated_cost:.2f}")
    
    proceed = input("\nDo you want to proceed? (y/n): ")
    if proceed.lower() != 'y':
        logger.info("Aborting...")
        return
    
    # Process in batches with rate limiting and better error handling
    BATCH_SIZE = 5
    RATE_LIMIT_PAUSE = 5
    MAX_RETRIES = 3
    total_items = len(raw_catalog)
    
    logger.info(f"Processing {total_items} catalog items...")
    for i in range(0, total_items, BATCH_SIZE):
        batch = raw_catalog[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(total_items + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        retries = 0
        while retries < MAX_RETRIES:
            try:
                processed_items = processor.process_catalog(batch, verbose=False)
                logger.info(f"Successfully processed {len(processed_items)} items in current batch")
                time.sleep(RATE_LIMIT_PAUSE)
                break
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                retries += 1
                if "429" in str(e):  # Rate limit error
                    wait_time = 60 * retries  # Exponential backoff
                    logger.info(f"Rate limit hit, pausing for {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Non-rate-limit error: {str(e)}")
                    if retries == MAX_RETRIES:
                        logger.error(f"Max retries reached for batch {i//BATCH_SIZE + 1}, skipping...")
                    time.sleep(RATE_LIMIT_PAUSE)
    
    # Example search using the processed catalog
    search_examples = [
        {
            "query": "luxury designer handbags",
            "filters": {"price_tier": "luxury"}
        },
        {
            "query": "casual summer dresses",
            "filters": {"category": "dresses", "gender": "F"}
        }
    ]
    
    logger.info("\nTesting search functionality:")
    for example in search_examples:
        logger.info(f"\nSearching for: {example['query']}")
        results = processor.search_catalog(
            query_text=example['query'],
            filters=example['filters']
        )
        
        # Display results
        if isinstance(results, dict) and 'documents' in results and 'metadatas' in results:
            # Handle the original dictionary format
            for idx, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                logger.info(f"\nResult {idx + 1}:")
                if isinstance(metadata, dict):
                    logger.info(f"Brand: {metadata.get('brand', 'N/A')}")
                    logger.info(f"Category: {metadata.get('category', 'N/A')}")
                    logger.info(f"URL: {metadata.get('url', 'N/A')}")
                elif isinstance(metadata, list) and metadata:
                    first_item = metadata[0]
                    logger.info(f"Brand: {first_item.get('brand', 'N/A')}")
                    logger.info(f"Category: {first_item.get('category', 'N/A')}")
                    logger.info(f"URL: {first_item.get('url', 'N/A')}")
                logger.info(f"Description preview: {doc[:200]}...")
        elif isinstance(results, list):
            # Handle list format
            for idx, result in enumerate(results):
                logger.info(f"\nResult {idx + 1}:")
                if isinstance(result, dict):
                    logger.info(f"Brand: {result.get('brand', 'N/A')}")
                    logger.info(f"Category: {result.get('category', 'N/A')}")
                    logger.info(f"URL: {result.get('url', 'N/A')}")
                    logger.info(f"Description preview: {str(result.get('description', ''))[:200]}...")
    
    logger.info("\nSearching for: luxury designer handbags")
    results = processor.search("luxury designer handbags")
    logger.info("\nResult 1:")
    
    # Fix for handling list results
    if results:
        for idx, result in enumerate(results, 1):
            if isinstance(result, dict):
                logger.info(f"Brand: {result.get('brand', 'N/A')}")
                logger.info(f"Title: {result.get('title', 'N/A')}")
                logger.info(f"Description: {result.get('description', 'N/A')}")
            elif isinstance(result, list) and result:
                first_item = result[0]
                if isinstance(first_item, dict):
                    logger.info(f"Brand: {first_item.get('brand', 'N/A')}")
                    logger.info(f"Title: {first_item.get('title', 'N/A')}")
                    logger.info(f"Description: {first_item.get('description', 'N/A')}")
            logger.info("---")

if __name__ == "__main__":
    main() 
