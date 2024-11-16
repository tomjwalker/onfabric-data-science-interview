from src.data_processing.embeddings import EmbeddingManager
from src.config.model_config import ModelConfig, VectorDBConfig

def test_embeddings():
    manager = EmbeddingManager(ModelConfig(), VectorDBConfig())
    test_text = "This is a test sentence"
    embedding = manager.create_embedding(test_text)
    print(f"Embedding dimensions: {len(embedding)}")

if __name__ == "__main__":
    test_embeddings() 