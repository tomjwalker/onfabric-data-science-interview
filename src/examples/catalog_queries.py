"""
Example queries and operations with processed catalog data for a luxury fashion recommendation system.

This module provides functionality for building and executing personalised catalog queries
based on user preferences and fashion analysis data. It includes:

Classes:
    UserPreferenceQueryBuilder: Builds weighted search queries based on user preferences
        and fashion trend analysis.

Functions:
    get_personalized_recommendations: Generates personalised product recommendations
        using the query builder and catalog processor.
    demonstrate_catalog_queries: Provides example usage of various catalog search
        operations.

The module supports different types of searches including:
- User preference-based recommendations
- Category-specific product searches
- Brand-specific product searches
- Occasion-based recommendations

Dependencies:
    - src.data_processing.catalog_processor
    - src.config.model_config
"""
from src.data_processing.catalog_processor import CatalogProcessor
from src.config.model_config import ModelConfig, VectorDBConfig
from typing import Dict, List, Any
from collections import Counter
import json

class UserPreferenceQueryBuilder:
    def __init__(self, fashion_analysis: Dict, user_analysis: str):
        self.fashion_data = fashion_analysis
        self.user_analysis = user_analysis
        self.brand_weights = self._calculate_brand_weights()
        self.product_weights = self._calculate_product_weights()
        
    def _calculate_brand_weights(self) -> Dict[str, float]:
        """Calculate normalized weights for brands based on search frequency"""
        luxury_brands = self.fashion_data["entity_frequencies"]["brands"]["luxury"]
        total_searches = sum(luxury_brands.values())
        return {brand: count/total_searches for brand, count in luxury_brands.items()}
    
    def _calculate_product_weights(self) -> Dict[str, float]:
        """Calculate normalized weights for product types"""
        product_types = self.fashion_data["entity_frequencies"]["product_types"]
        flattened_products = {}
        for category in product_types.values():
            flattened_products.update(category)
        total = sum(flattened_products.values())
        return {prod: count/total for prod, count in flattened_products.items()}
    
    def generate_weighted_queries(self, n_queries: int = 3) -> List[Dict[str, Any]]:
        """Generate multiple weighted queries based on user preferences"""
        queries = []
        
        # Top brands query - more flexible
        top_brands = sorted(self.brand_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        brand_query = {
            "query_text": f"luxury fashion items by {top_brands[0][0]}",
            # No filters, rely on query text for brand matching
            "n_results": 5
        }
        print(f"Debug - Brand Query: {brand_query}")  # Debug print
        queries.append(brand_query)
        
        # Top product types query - more flexible
        top_products = sorted(self.product_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        product_query = {
            "query_text": f"luxury {top_products[0][0]}",  # Include product type in query
            "filters": {
                "price_tier": "luxury"  # Only filter on price tier
            },
            "n_results": 5
        }
        print(f"Debug - Product Query: {product_query}")  # Debug print
        queries.append(product_query)
        
        # General luxury query (this one works, keep it as is)
        general_query = {
            "query_text": "luxury fashion items",
            "filters": {
                "price_tier": "luxury"
            },
            "n_results": 5
        }
        print(f"Debug - General Query: {general_query}")  # Debug print
        queries.append(general_query)
        
        return queries

def get_personalized_recommendations(
    processor: CatalogProcessor,
    fashion_analysis: Dict,
    user_analysis: str
) -> Dict[str, List]:
    """Get personalized product recommendations"""
    query_builder = UserPreferenceQueryBuilder(fashion_analysis, user_analysis)
    queries = query_builder.generate_weighted_queries()
    
    results = {}
    for i, query in enumerate(queries):
        try:
            print(f"\nTrying recommendation set {i+1}...")  # Debug print
            query_results = processor.search_catalog(**query)
            if query_results and query_results.get('documents'):
                print(f"Found {len(query_results['documents'][0])} results")  # Debug print
                results[f"recommendation_set_{i+1}"] = query_results
            else:
                print(f"No results found for recommendation set {i+1}")
        except Exception as e:
            print(f"Error processing recommendation set {i+1}: {str(e)}")
    
    return results

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
    import argparse
    
    parser = argparse.ArgumentParser(description='Luxury fashion catalog query tool')
    parser.add_argument('--fashion-analysis', 
                       default='data/processed/fashion_analysis.json',
                       help='Path to fashion analysis JSON file')
    parser.add_argument('--user-analysis',
                       default='data/processed/user_comp_analysis.txt',
                       help='Path to user analysis text file')
    parser.add_argument('--output',
                       default=None,
                       help='Output file path for recommendations (optional)')
    parser.add_argument('--demo',
                       action='store_true',
                       help='Run demonstration queries instead of personalised recommendations')
    
    args = parser.parse_args()
    
    # Initialize processor
    model_config = ModelConfig()
    db_config = VectorDBConfig()
    processor = CatalogProcessor(
        model_config=model_config,
        db_config=db_config
    )
    
    if args.demo:
        results = demonstrate_catalog_queries()
    else:
        # Load fashion analysis data
        with open(args.fashion_analysis, 'r') as f:
            fashion_analysis = json.load(f)

        # Load user analysis
        with open(args.user_analysis, 'r') as f:
            user_analysis = f.read()

        # Get personalized recommendations
        results = get_personalized_recommendations(
            processor=processor,
            fashion_analysis=fashion_analysis,
            user_analysis=user_analysis
        )
    
    # Output handling
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    else:
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