"""
Main module for processing fashion catalog data
"""
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
import chromadb
from .schema import CatalogItem, ProcessedCatalogItem
from .entity_extractor import EntityExtractor
from .embeddings import EmbeddingManager
from ..config.model_config import ModelConfig, VectorDBConfig
import os
from chromadb.utils import embedding_functions
import tiktoken

class CatalogProcessor:
    def __init__(
        self,
        model_config: ModelConfig = ModelConfig(),
        db_config: VectorDBConfig = VectorDBConfig()
    ):
        self.entity_extractor = EntityExtractor(model_config)
        self.embedding_manager = EmbeddingManager(model_config, db_config)
        
        # Create the embedding function without specifying dimensions
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv('OPENAI_API_KEY'),
            model_name="text-embedding-ada-002"
        )
        
        self.chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        
        # Use the same embedding function for the collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=db_config.COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.model_config = model_config
    
    def estimate_processing_cost(self, catalog_data: List[Dict[str, Any]]) -> Tuple[int, float]:
        """
        Estimate the tokens and cost for processing the catalog data.
        
        Returns:
            Tuple[int, float]: (estimated total tokens, estimated total cost in USD)
        """
        # Sample the first item or use a small batch for estimation
        sample_size = min(5, len(catalog_data))
        sample_items = catalog_data[:sample_size]
        
        # Get the encoding based on the model
        encoding = tiktoken.encoding_for_model(ModelConfig.DEFAULT_MODEL)
        
        # Estimate tokens for a sample item
        sample_tokens = 0
        for item in sample_items:
            # Convert to string to estimate tokens for the main content
            item_str = f"{item.get('title', '')} {item.get('description', '')} {item.get('brand', '')}"
            tokens = len(encoding.encode(item_str))
            sample_tokens += tokens
        
        # Calculate average tokens per item
        avg_tokens_per_item = sample_tokens / sample_size
        
        # Estimate total tokens for all items
        total_items = len(catalog_data)
        estimated_total_tokens = avg_tokens_per_item * total_items
        
        # Add overhead for entity extraction and embedding (approximately 1.5x)
        estimated_total_tokens *= 1.5
        
        # Get cost rates from model config
        input_cost, output_cost = ModelConfig.get_cost_rates(ModelConfig.DEFAULT_MODEL)
        
        # Estimate total cost (assuming roughly equal input/output tokens)
        total_cost = (estimated_total_tokens * (input_cost + output_cost) / 2000)  # Divide by 2000 as rates are per 1k tokens
        
        return int(estimated_total_tokens), total_cost

    def process_catalog(self, catalog_data: List[Dict[str, Any]], verbose: bool = False) -> List[ProcessedCatalogItem]:
        """
        Process entire catalog and store in vector database
        """
        if verbose:
            estimated_tokens, estimated_cost = self.estimate_processing_cost(catalog_data)
            print("Estimated processing requirements:")
            print(f"Total items: {len(catalog_data)}")
            print(f"Estimated tokens: {estimated_tokens:,}")
            print(f"Estimated cost: ${estimated_cost:.2f}")
            
            proceed = input("Do you want to proceed? (y/n): ")
            if proceed.lower() != 'y':
                print("Aborting...")
                return []
        
        processed_items = []
        
        for raw_item in catalog_data:
            try:
                # Convert raw dict to Pydantic model and immediately to dict
                catalog_item = CatalogItem(**raw_item).model_dump()
                
                # Extract entities (modify entity_extractor to accept dict instead of CatalogItem)
                entities = self.entity_extractor.extract_entities(catalog_item)
                
                # Process item and generate embedding
                processed_item = self.embedding_manager.process_catalog_item(
                    catalog_item,
                    entities
                )
                
                processed_items.append(processed_item)
                
                # Add to vector database
                self.collection.add(
                    documents=[processed_item.searchable_text],
                    embeddings=[processed_item.embedding],
                    metadatas=[{
                        "brand": entities.brand,
                        "category": entities.category,
                        "gender": catalog_item.get('gender'),  # Changed to dict access
                        "price_tier": entities.price_tier,
                        "url": str(catalog_item.get('product_url'))  # Changed to dict access
                    }],
                    ids=[processed_item.id]
                )
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
        
        return processed_items

    def search_catalog(
        self, 
        query_text: str, 
        filters: Optional[Dict] = None,
        n_results: int = 5
    ) -> Dict:
        where_filter = None
        if filters:
            # Format multiple conditions using $and
            conditions = []
            for key, value in filters.items():
                conditions.append({key: {"$eq": value}})
            
            # If we have multiple conditions, use $and, otherwise use the single condition
            if len(conditions) > 1:
                where_filter = {"$and": conditions}
            elif conditions:
                where_filter = conditions[0]

        # Use the same embedding function for querying
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )