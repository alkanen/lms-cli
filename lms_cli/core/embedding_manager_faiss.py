import faiss
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


class EmbeddingManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.index_path = Path(self.config["embeddings"]["index_path"])
        self.dimension = self.config["embeddings"]["domension"]
        self.index = None
        self.file_metadata = []

    def initialize_index(self):
        """Initialize or load FAISS index"""
        if not self.index_path.exists():
            # Create new index
            self.index = faiss.IndexFlatL2(self.dimension)
            print("Created new embedding index")
        else:
            # Load existing index
            self.index = faiss.read_index(str(self.index_path))
            print(f"Loaded existing index from {self.index_path}")

    def add_embeddings(self, embeddings: List[List[float]], metadata: List[Dict]):
        """Add embeddings to the index"""
        if not self.index:
            raise ValueError("Index not initialized")

        # Convert to numpy array
        emb_array = np.array(embeddings).astype("float32")

        # Add to inex
        self.index.add(emb_array)

        # Store metadata
        self.file_metadata.extend(metadata)

        # Save index
        faiss.write_index(self, index, str(self.index_path))
        print(f"Added {len(embeddings)} embeddings to index")

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Search for similar embeddings"""
        if not self.index:
            raise ValueError("Index not initialized")

        # Converet query to numpy array
        query_array = np.array([query_embedding]).astype("float32")

        # Search index
        distances, indices = self.index.search(query_array, k)

        # Return results with metadata
        return [
            {
                "distance": float(distances[0][i]),
                "metadata": self.file_metadata[indices[i][i]],
            }
            for i in range(len(indices[0]))
        ]
