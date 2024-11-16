from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict
from enum import Enum

class Gender(str, Enum):
    MALE = "M"
    FEMALE = "F"

class CatalogItem(BaseModel):
    main_image: str = Field(alias='MAIN_IMAGE')
    second_image: Optional[HttpUrl] = None
    third_image: Optional[HttpUrl] = None
    fourth_image: Optional[HttpUrl] = None
    product_url: HttpUrl = Field(alias="LYST_PRODUCT_URL")
    gender: Gender = Field(alias="GENDER")
    category: str = Field(alias="CATEGORY")
    short_description: str = Field(alias="SHORT_DESCRIPTION")
    long_description: str = Field(alias="LONG_DESCRIPTION")

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True

class ExtractedEntities(BaseModel):
    brand: Optional[str] = None
    category: str
    style_descriptors: List[str] = Field(default_factory=list)
    price_tier: Optional[str] = None
    season: Optional[str] = None
    materials: List[str] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)

class ProcessedCatalogItem(BaseModel):
    id: str
    raw_item: CatalogItem
    entities: ExtractedEntities
    embedding: List[float]
    searchable_text: str 