"""
Configuration settings for AI models and processing parameters
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Central configuration for OpenAI model settings"""
    
    DEFAULT_MODEL = "gpt-4"  # Changed from instance variable to class variable
    
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
    
    ENTITY_EXTRACTION_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
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