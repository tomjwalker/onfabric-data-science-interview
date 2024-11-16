"""
Module for generating and managing embeddings for fashion catalog items
"""
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from .schema import ProcessedCatalogItem, ExtractedEntities
from ..config.model_config import ModelConfig, VectorDBConfig
from openai import OpenAI
import os

class EmbeddingManager:
    def __init__(
        self,
        model_config: ModelConfig = ModelConfig(),
        db_config: VectorDBConfig = VectorDBConfig()
    ):
        self.model_config = model_config
        self.db_config = db_config
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=client.api_key,
            model_name=model_config.EMBEDDING_MODEL
        )
        
    def generate_searchable_text(
        self,
        item: Dict[str, Any],
        entities: ExtractedEntities
    ) -> str:
        """
        Generate rich text description for embedding
        """
        return f"""
        {item.get('short_description', '')}
        {item.get('long_description', '')}
        Brand: {entities.brand}
        Category: {entities.category}
        Style: {', '.join(entities.style_descriptors)}
        Materials: {', '.join(entities.materials)}
        Colors: {', '.join(entities.colors)}
        Season: {entities.season}
        Price Tier: {entities.price_tier}
        """.strip()
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for searchable text
        """
        return self.embedding_function([text])[0]
    
    def process_catalog_item(
        self,
        item: Dict[str, Any],
        entities: ExtractedEntities
    ) -> ProcessedCatalogItem:
        """
        Process a single catalog item into a format ready for vector storage
        """
        searchable_text = self.generate_searchable_text(item, entities)
        embedding = self.create_embedding(searchable_text)
        
        return ProcessedCatalogItem(
            id=str(hash(item.get('product_url', ''))),
            raw_item=item,
            entities=entities,
            embedding=embedding,
            searchable_text=searchable_text
        ) 