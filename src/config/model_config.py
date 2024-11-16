"""
Configuration settings for AI models and processing parameters
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    ENTITY_EXTRACTION_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.3
    
    # Prompt templates
    ENTITY_EXTRACTION_TEMPLATE: str = """
    Extract the following entities from this product listing:
    - Brand name
    - Product category
    - Style descriptors (e.g. casual, formal, luxury)
    - Price indicators (from description/URL)
    - Season indicators
    - Materials
    - Colors
    
    Product data: {item}
    
    Return as JSON with these exact keys: brand, category, style_descriptors, price_tier, season, materials, colors
    """

    ENTITY_EXTRACTION_SYSTEM_PROMPT: str = """
    You are a fashion catalog item analyzer. Extract the following attributes from the product description:
    - brand: The brand name of the product
    - category: The product category (e.g., dress, handbag, shoes)
    - materials: List of materials used in the product
    - price_tier: Either 'budget', 'mid-range', or 'luxury'
    - gender: 'M' for men, 'F' for women, 'U' for unisex
    
    Return the information in JSON format.
    """

@dataclass
class VectorDBConfig:
    COLLECTION_NAME: str = "fashion_catalog"
    DISTANCE_METRIC: str = "cosine"
    EMBEDDING_DIMENSION: int = 1536  # For text-embedding-ada-002 