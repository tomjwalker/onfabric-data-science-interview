"""
Main module for processing fashion catalog data
"""
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import chromadb
from .schema import CatalogItem, ProcessedCatalogItem
from .entity_extractor import EntityExtractor
from .embeddings import EmbeddingManager
from ..config.model_config import ModelConfig, VectorDBConfig
import os
from chromadb.utils import embedding_functions

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
    
    def process_catalog(self, catalog_data: List[Dict[str, Any]]) -> List[ProcessedCatalogItem]:
        """
        Process entire catalog and store in vector database
        """
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