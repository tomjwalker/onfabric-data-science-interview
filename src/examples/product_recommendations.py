"""
Personalised fashion catalog query and recommendation system.

This module combines user behaviour analysis with vector-based catalog search to generate
tailored fashion recommendations. It processes pre-analyzed user data and searches a
vector database of fashion items using semantic queries.

Classes:
    UserPreferenceQueryBuilder:
        Builds weighted search queries by combining multiple data sources:
        1. Statistical Analysis:
           - Brand preferences (from fashion analysis)
           - Product type preferences (from fashion analysis)
        2. User Behaviour Analysis:
           - Core style identity
           - Style preferences
           - Aesthetic preferences

        Methods:
        - generate_weighted_queries(): Orchestrates query generation from all sources
        - generate_fashion_analysis_queries(): Creates statistical-based queries
        - generate_user_analysis_query(): Creates behaviour-based query
        - _calculate_brand_weights(): Normalizes brand frequency data
        - _calculate_product_weights(): Normalizes product type frequency data

Functions:
    get_personalized_recommendations(
        processor: CatalogProcessor,
        fashion_analysis: Dict,
        user_analysis: str
    ) -> Dict[str, List]:
        Generates three sets of personalised product recommendations:
        1. Brand-based recommendations from statistical analysis
        2. Product-based recommendations from statistical analysis
        3. Style-based recommendations from user behaviour analysis

Command-line Usage:
    python utils/generate_tree.py [--path PATH] [--desc] [--output PATH]

    Examples:
        # Basic usage (current directory)
        python utils/generate_tree.py

        # Most common usage - save to project structure file
        python utils/generate_tree.py --output utils/project_structure.txt

        # Include file descriptions
        python utils/generate_tree.py --desc

        # Specify different starting path with descriptions
        python utils/generate_tree.py --path src/examples --desc

        # Full example with all options
        python utils/generate_tree.py --path /path/to/project --desc --output tree_output.txt

Input Files:
    fashion_analysis.json: Contains brand and product type frequency analysis
    user_comp_analysis.txt: Contains processed user behaviour analysis in markdown format
    Vector DB: Pre-embedded fashion catalog items (via ChromaDB)

Output:
    JSON-formatted recommendations containing:
    - Product documents
    - Product metadata (brand, category, URL)
    - Relevance scores

Dependencies:
    Internal:
        - src.data_processing.catalog_processor.CatalogProcessor
        - src.config.model_config.{ModelConfig, VectorDBConfig}
    External:
        - typing
        - json
        - collections
        - argparse
        - re
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
    
    def generate_fashion_analysis_queries(self) -> List[Dict[str, Any]]:
        """Generate queries based on fashion analysis data"""
        queries = []
        
        # Top brands query
        top_brands = sorted(self.brand_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        brand_query = {
            "query_text": f"luxury fashion items by {top_brands[0][0]}",
            "n_results": 5
        }
        queries.append(brand_query)
        
        # Top product types query
        top_products = sorted(self.product_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        product_query = {
            "query_text": f"luxury {top_products[0][0]}",
            "filters": {"price_tier": "luxury"},
            "n_results": 5
        }
        queries.append(product_query)
        
        return queries

    def generate_user_analysis_query(self) -> Dict[str, Any]:
        """Generate query based on user behaviour analysis"""
        import re
        
        # Extract key information from markdown sections
        sections = re.split(r'###\s+\d+\.\s+', self.user_analysis)
        
        # Find Core Style Identity section
        style_section = next((s for s in sections if 'Core Style Identity' in s), '')
        
        # Extract primary style categories and aesthetic preferences
        style_lines = [line.strip('- *') for line in style_section.split('\n') 
                      if line.strip().startswith('-')]
        
        # Combine relevant style information into a query
        query_text = "luxury fashion items with "
        if style_lines:
            # Take first two style descriptors to keep query focused
            descriptors = [line.split(':')[1].strip() if ':' in line else line 
                         for line in style_lines[:2]]
            query_text += " and ".join(descriptors)
        
        return {
            "query_text": query_text,
            "filters": {"price_tier": "luxury"},
            "n_results": 5
        }

    def generate_weighted_queries(self) -> List[Dict[str, Any]]:
        """Orchestrate generation of all recommendation queries"""
        queries = self.generate_fashion_analysis_queries()
        queries.append(self.generate_user_analysis_query())
        
        # Debug prints
        for i, query in enumerate(queries, 1):
            print(f"Debug - Query {i}: {query}")
        
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Luxury fashion product recommendation tool')
    parser.add_argument('--fashion-analysis', 
                       default='data/processed/fashion_analysis.json',
                       help='Path to fashion analysis JSON file')
    parser.add_argument('--user-analysis',
                       default='data/processed/user_comp_analysis.txt',
                       help='Path to user analysis text file')
    parser.add_argument('--output',
                       default='data/processed/product_recommendations.json',
                       help='Output file path for recommendations (default: data/processed/product_recommendations.json)')
    
    args = parser.parse_args()
    
    # Initialize processor
    model_config = ModelConfig()
    db_config = VectorDBConfig()
    processor = CatalogProcessor(
        model_config=model_config,
        db_config=db_config
    )
    
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