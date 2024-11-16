"""
Module for extracting structured entities from fashion catalog items
"""
from typing import Dict, Any
import openai
import json
from ..config.model_config import ModelConfig
from .schema import CatalogItem, ExtractedEntities

class EntityExtractor:
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        
    def extract_entities(self, catalog_item: Dict) -> ExtractedEntities:
        try:
            # Convert the catalog item to a JSON-serializable dict
            serializable_item = {}
            for key, value in catalog_item.items():
                # Convert URL objects and any other objects to strings
                serializable_item[key] = str(value) if hasattr(value, '__str__') else value

            # Create a structured prompt
            prompt = f"""Please analyze this product and extract the following attributes:
            - brand
            - category
            - materials (as a list)
            - price_tier (luxury, premium, or mass-market)
            - gender (mens, womens, or unisex)

            Product details:
            {json.dumps(serializable_item, indent=2)}

            Respond with a JSON object only."""

            response = openai.chat.completions.create(
                model=self.config.ENTITY_EXTRACTION_MODEL,
                messages=[
                    {"role": "system", "content": self.config.ENTITY_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=150,
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            # Get the response content
            result = response.choices[0].message.content
            if not result:
                raise ValueError("Empty response from OpenAI")
            
            # Parse the JSON response
            parsed_result = json.loads(result)
            
            # Ensure materials is a list
            if isinstance(parsed_result.get('materials'), str):
                parsed_result['materials'] = [parsed_result['materials']]
            elif not parsed_result.get('materials'):
                parsed_result['materials'] = []
            
            return ExtractedEntities(**parsed_result)
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}, Response was: {result}")
            return ExtractedEntities(
                brand="unknown",
                category="unknown",
                materials=[],
                price_tier="unknown",
                gender="unknown"
            )
        except Exception as e:
            print(f"Error extracting entities: {str(e)}")
            return ExtractedEntities(
                brand="unknown",
                category="unknown",
                materials=[],
                price_tier="unknown",
                gender="unknown"
            ) 