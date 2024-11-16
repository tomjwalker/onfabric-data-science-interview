"""
Example queries and operations with processed catalog
"""
from src.data_processing.catalog_processor import CatalogProcessor
from src.config.model_config import ModelConfig, VectorDBConfig

def demonstrate_catalog_queries():
    # Initialize with explicit configurations
    model_config = ModelConfig()
    db_config = VectorDBConfig()
    processor = CatalogProcessor(
        model_config=model_config,
        db_config=db_config
    )
    
    # Example 1: Search for products similar to user preferences
    user_preferences = """
    Luxury brand preferences including Tory Burch and Valentino.
    Interest in designer sandals and handbags.
    Preference for summer styles and office wear.
    """
    
    results = processor.search_catalog(
        query_text=user_preferences,
        filters={"gender": "F", "price_tier": "luxury"},
        n_results=5
    )
    
    # Example 2: Category-specific search
    handbag_search = processor.search_catalog(
        query_text="designer leather handbags suitable for office",
        filters={
            "category": "bags",
            "price_tier": "luxury"
        },
        n_results=3
    )
    
    # Example 3: Brand-specific search
    tory_burch_items = processor.search_catalog(
        query_text="casual comfortable sandals",
        filters={
            "brand": "tory burch",
            "category": "shoes"
        },
        n_results=3
    )
    
    return {
        "user_preferences_results": results,
        "handbag_results": handbag_search,
        "tory_burch_results": tory_burch_items
    }

if __name__ == "__main__":
    results = demonstrate_catalog_queries()
    
    # Print results in a formatted way
    for query_type, query_results in results.items():
        print(f"\n=== {query_type} ===")
        if not query_results.get('documents'):
            print("No results found")
            continue
            
        for idx, (doc, metadata) in enumerate(zip(
            query_results['documents'][0],  # Access first list element
            query_results['metadatas'][0]   # Access first list element
        )):
            print(f"\nResult {idx + 1}:")
            print(f"Brand: {metadata['brand']}")
            print(f"Category: {metadata['category']}")
            print(f"URL: {metadata['url']}")
            print(f"Description preview: {doc[:150]}...") 