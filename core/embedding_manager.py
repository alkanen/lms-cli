import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

"""Index format:
{
    filename: {
        "vector": [0.0, 1.0, ...],
        "metadata": {}
    },
    ...
}
"""


class EmbeddingManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.index_path = Path(self.config["embeddings"]["index_path"])
        self.dimension = self.config["embeddings"]["dimension"]
        self.inclusion_paths = set(self.config["embeddings"]["include_paths"])
        self.exclusion_paths = set(self.config["embeddings"]["exclude_paths"])
        self.index = {}

    def initialize_index(self):
        """Initialize or load naïve index"""
        self.index = {}

        if not self.index_path.exists():
            # Create new index
            print("Created new embedding index")
        else:
            # Load existing index
            with self.index_path.open() as f:
                data = json.load(f)
                for filename in data:
                    vector = np.array(data[filename]["vector"]).astype("float32")
                    metadata = data[filename]["metadata"]
                    self.index[filename] = {"vector": vector, "metadata": metadata}

            print(f"Loaded existing index from '{self.index_path}'")

    def delete_with_metadata_key_value(self, metadata: Dict):
        """Delete item from index where all the keys in metadata match the indexed
        metadata.  Keys not in argument will not be compared."""

        to_delete = []

        for filename in self.index:
            is_match = True
            fm = self.index[filename]["metadata"]

            for key in metadata:
                if key not in fm:
                    is_match = False
                    break

                if fm[key] != metadata[key]:
                    is_match = False
                    break

            if is_match:
                to_delete.append(filename)

        for filename in to_delete:
            del self.index[filename]

    def add_embeddings(self, embeddings: List[List[float]], metadata: List[Dict]):
        """Add embeddings to the index"""
        # If file already exists in database, check if it has changed since last time

        count = 0

        for i in range(len(metadata)):
            filename = metadata[i]["file"]

            file_modified = datetime.fromtimestamp(os.path.getmtime(filename))

            if filename in self.index:
                add_time = self.index[filename]["metadata"].get("timestamp")
                if (
                    add_time
                    and datetime.strptime(add_time, "%Y-%m-%dT%H:%M:%S")
                    >= file_modified
                ):
                    # print(f"File '{filename}' not changed, skipping")
                    continue

            # Convert to numpy arrays
            emb_array = np.array(embeddings[i]).astype("float32")

            # Add to index
            self.index[filename] = {
                "vector": emb_array,
                "metadata": {
                    **metadata[i],
                    "timestamp": datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S"),
                },
            }

            count += 1
            print(f"Adding embeddings for '{filename}'")

        # Save index
        with self.index_path.open(mode="w") as f:
            d = {}

            for filename in self.index:
                v = self.index[filename]["vector"].tolist()
                m = self.index[filename]["metadata"]
                d[filename] = {"vector": v, "metadata": m}

            json.dump(d, fp=f, indent=4)

        print(f"Added {count} new embeddings to index")

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Search for similar embeddings"""
        if not self.index:
            raise ValueError("Index not initialized")

        # Converet query to numpy array
        query_array = (
            np.array([query_embedding])
            .astype("float32")
            .reshape((len(query_embedding),))
        )

        # Search index
        found = []
        for filename in self.index:
            v = self.index[filename]["vector"]
            similarity = np.dot(v, query_array)

            # If arrays aren't full, simply append values, otherwise compare and add if
            # better than the worst
            if len(found) < k:
                found.append({"similarity": similarity, "filename": filename})
                found = sorted(found, key=lambda x: x["similarity"], reverse=True)

            else:
                worse_idx = -1
                for idx, data in enumerate(found):
                    if data["similarity"] < similarity:
                        worse_idx = idx
                        break

                if worse_idx >= 0:
                    found.insert(
                        worse_idx, {"similarity": similarity, "filename": filename}
                    )
                    found.pop(-1)

        # Return results with metadata
        return [
            {
                "similarity": float(f["similarity"]),
                "metadata": self.index[f["filename"]]["metadata"],
            }
            for f in found
        ]
