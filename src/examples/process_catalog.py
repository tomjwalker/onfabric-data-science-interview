"""
Example script demonstrating how to use the catalog processing modules
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

def load_catalog(catalog_path: Path) -> List[Dict]:
    """Load the fashion catalog from JSON file"""
    try:
        with open(catalog_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert to CatalogItem and immediately to dict
            return [CatalogItem.parse_obj(item).dict() for item in data]
    except Exception as e:
        logger.error(f"Error loading catalog file: {str(e)}")
        raise

def main(verbose: bool, test_mode: bool):
    # Initialize configurations
    model_config = ModelConfig()
    db_config = VectorDBConfig()
    
    # Initialize the catalog processor
    processor = CatalogProcessor(model_config, db_config)
    
    # Load catalog data
    catalog_path = Path("data/raw/fashion_catalog.json")
    logger.info(f"Loading catalog from {catalog_path}")
    raw_catalog = load_catalog(catalog_path)
    
    # Get current count of processed items
    current_count = processor.collection.count()
    logger.info(f"Found {current_count} existing items in database")
    
    # Skip already processed items
    if current_count > 0:
        raw_catalog = raw_catalog[current_count:]
        logger.info(f"Skipping {current_count} already processed items")
    
    # Process in batches with rate limiting and better error handling
    BATCH_SIZE = 5  # Even smaller batch size
    RATE_LIMIT_PAUSE = 5  # Longer pause between batches
    MAX_RETRIES = 3
    total_items = len(raw_catalog)
    
    logger.info(f"Processing {total_items} catalog items...")
    
    for i in range(0, total_items, BATCH_SIZE):
        batch = raw_catalog[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(total_items + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        retries = 0
        while retries < MAX_RETRIES:
            try:
                processed_items = processor.process_catalog(batch)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process fashion catalog data')
    parser.add_argument('--test', action='store_true', help='Run in test mode (process only 2 chunks)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
    args = parser.parse_args()
    
    print(f"Running with test_mode={args.test}, verbose={args.verbose}")
    main(verbose=args.verbose, test_mode=args.test) 