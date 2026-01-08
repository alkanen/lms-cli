import json
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import yaml

"""Index format:
[
    {
        "vector": [0.0, 1.0, ...],
        "metadata": {}
    },
    ...
]
"""


class EmbeddingManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.index_path = Path(self.config["embeddings"]["index_path"])
        self.dimension = self.config["embeddings"]["dimension"]
        self.index = []
        self.file_metadata = []

        self.initialize_index()

    def initialize_index(self):
        """Initialize or load naïve index"""
        self.index = []
        self.file_metadata = []

        if not self.index_path.exists():
            # Create new index
            print("Created new embedding index")
        else:
            # Load existing index
            with self.index_path.open() as f:
                data = json.load(f)
                for d in data:
                    self.file_metadata.append(d["metadata"])
                    self.index.append(np.array(d["vector"]).astype("float32"))

            print(f"Loaded existing index from {self.index_path}")

    def delete_with_metadata_key_value(self, metadata: Dict):
        """Delete item from index where all the keys in metadata match the indexed
        metadata.  Keys not in argument will not be compared."""
        indices_to_delete = []

        for i, fm in zip(self.index, self.file_metadata):
            is_match = True
            for key in metadata:
                if key not in fm:
                    is_match = False
                    break

                if fm[key] != metadata[key]:
                    is_match = False
                    break

            if is_match:
                indices_to_delete.append(i)

        for i in reversed(indices_to_delete):
            del self.index[i]
            del self.file_metadata[i]

    def add_embeddings(self, embeddings: List[List[float]], metadata: List[Dict]):
        """Add embeddings to the index"""
        # if self.index:
        #     raise ValueError("Index not initialized")

        # Convert to numpy arrays
        emb_array = [np.array(emb).astype("float32") for emb in embeddings]

        # Add to index
        self.index.extend(emb_array)

        # Store metadata
        self.file_metadata.extend(metadata)

        # Save index
        with self.index_path.open(mode="w") as f:
            d = []

            for v, m in zip(self.index, self.file_metadata):
                d.append(
                    {"vector": v.tolist(), "metadata": m}
                )

            json.dump(d, fp=f, indent=4)
        print(f"Added {len(embeddings)} embeddings to index")

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Search for similar embeddings"""
        if not self.index:
            raise ValueError("Index not initialized")

        # Converet query to numpy array
        query_array = np.array([query_embedding]).astype("float32")

        # Search index
        found = []
        for i, v in enumerate(self.index):
            dist = np.dot(v, query_array)

            # If arrays aren't full, simply append values, otherwise compare and add if
            # better than the worst
            if len(found) < k:
                found.append({"distance":dist, "index": i})
                found = sorted(found, key=lambda x: x["distance"])

            else:
                worse_idx = -1
                for idx, data in enumerate(found):
                    if data["distance"] > dist:
                        worse_idx = idx
                    else:
                        break

                if worse_idx >= 0:
                    found.insert(worse_idx, {"distance":dist, "index": i})
                    found.pop(-1)

        # Return results with metadata
        return [
            {
                "distance": float(f["distance"]),
                "metadata": self.file_metadata[f["index"]],
            }
            for f in found
        ]
