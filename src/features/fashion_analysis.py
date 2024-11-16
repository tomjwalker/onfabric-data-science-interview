from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass, field
import openai
from datetime import datetime
import time
import random

# Configuration constants
ENTITY_EXTRACTION_TEMPERATURE = 0.3
SUMMARY_TEMPERATURE = 0.7

@dataclass
class FashionAnalyzer:
    fashion_entities: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {
        "brands": {
            "luxury": [],
            "high_street": [],
            "sportswear": []
        },
        "product_types": {
            "clothing": [],
            "accessories": [],
            "footwear": []
        },
        "styles": {
            "aesthetic": [],
            "occasion": [],
            "seasonal": []
        }
    })
    
    fashion_topics: List[str] = field(default_factory=list)
    prompt_manager: Any = None
    chroma_client: Any = None
    model: str = "gpt-4o-mini"

    def analyze_chunk(self, chunk: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a chunk of browsing history for fashion insights.
        
        Args:
            chunk: List of dictionaries containing browsing history data
            
        Returns:
            Dictionary containing summary, entities, and topics
        """
        # Add small random delay to avoid rate limits
        time.sleep(random.uniform(1, 2))
        
        # Extract fashion entities
        entities = self._extract_fashion_entities(chunk)
        self._update_fashion_entities(entities)
        
        # Add small delay between API calls
        time.sleep(random.uniform(1, 2))
        
        # Extract topics
        topics = self._extract_topics(chunk)
        self._update_topics(topics)
        
        # Add small delay between API calls
        time.sleep(random.uniform(1, 2))
        
        # Generate summary focusing on fashion aspects
        summary = self._generate_chunk_summary(chunk, entities, topics)
        
        return {
            "summary": summary,
            "entities": entities,
            "topics": topics,
            "timestamp": datetime.now().isoformat()
        }

    def _extract_fashion_entities(self, chunk: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract fashion-specific entities using LLM.
        """
        try:
            prompt = self.prompt_manager.get_fashion_entity_prompt(
                data=json.dumps(chunk),
                current_entities=json.dumps(self.fashion_entities)
            )
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=ENTITY_EXTRACTION_TEMPERATURE,
                response_format={ "type": "json_object" }
            )
            
            entities = json.loads(response.choices[0].message.content)
            
            # Validate structure matches expected format
            self._validate_entity_structure(entities)
            
            return entities
            
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return self._empty_entity_structure()

    def _extract_topics(self, chunk: List[Dict[str, Any]]) -> List[str]:
        """
        Extract fashion-related topics using LLM.
        """
        try:
            prompt = self.prompt_manager.get_topic_extraction_prompt(
                data=json.dumps(chunk),
                current_topics=json.dumps(self.fashion_topics)
            )
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=ENTITY_EXTRACTION_TEMPERATURE,
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("topics", [])
            
        except Exception as e:
            print(f"Error in topic extraction: {e}")
            return []

    def _generate_chunk_summary(self, chunk: List[Dict[str, Any]], 
                              entities: Dict[str, Dict[str, List[str]]], 
                              topics: List[str]) -> str:
        """
        Generate a fashion-focused summary of the chunk.
        """
        try:
            context = {
                "chunk": chunk,
                "entities": entities,
                "topics": topics
            }
            
            prompt = self.prompt_manager.get_chunk_prompt(
                json.dumps(context)
            )
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=SUMMARY_TEMPERATURE
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in summary generation: {e}")
            return "Error generating summary"

    def _update_fashion_entities(self, new_entities: Dict[str, Dict[str, List[str]]]) -> None:
        """
        Update the fashion_entities dictionary with new entities.
        Allows duplicates to track frequency.
        """
        for category, subcategories in new_entities.items():
            for subcategory, entities in subcategories.items():
                if category in self.fashion_entities and subcategory in self.fashion_entities[category]:
                    self.fashion_entities[category][subcategory].extend(entities)

    def _update_topics(self, new_topics: List[str]) -> None:
        """
        Update topics list. Allows duplicates to track frequency.
        """
        self.fashion_topics.extend(new_topics)

    def get_entity_frequencies(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        Calculate frequencies of all entities.
        """
        frequencies = {}
        for category, subcategories in self.fashion_entities.items():
            frequencies[category] = {}
            for subcategory, entities in subcategories.items():
                frequencies[category][subcategory] = {
                    entity: entities.count(entity)
                    for entity in set(entities)
                }
        return frequencies

    def _validate_entity_structure(self, entities: Dict[str, Dict[str, List[str]]]) -> None:
        """
        Validate that the entity structure matches expected format.
        """
        expected_categories = {"brands", "product_types", "styles"}
        expected_subcategories = {
            "brands": {"luxury", "high_street", "sportswear"},
            "product_types": {"clothing", "accessories", "footwear"},
            "styles": {"aesthetic", "occasion", "seasonal"}
        }
        
        if not all(category in entities for category in expected_categories):
            raise ValueError("Missing expected categories in entity structure")
            
        for category, subcategories in expected_subcategories.items():
            if not all(subcategory in entities[category] for subcategory in subcategories):
                raise ValueError(f"Missing expected subcategories in {category}")

    def _empty_entity_structure(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Return an empty entity structure with correct format.
        """
        return {
            "brands": {
                "luxury": [],
                "high_street": [],
                "sportswear": []
            },
            "product_types": {
                "clothing": [],
                "accessories": [],
                "footwear": []
            },
            "styles": {
                "aesthetic": [],
                "occasion": [],
                "seasonal": []
            }
        }