import numpy as np
from typing import List, Dict
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

class Retriever:
    def __init__(self, index: faiss.Index, chunks_df: pd.DataFrame):
        self.index = index
        self.chunks = chunks_df

    def embed(self, query: str):
        v = EMBED_MODEL.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return v

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        qv = self.embed(query)
        D, I = self.index.search(qv, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            row = self.chunks.iloc[idx].to_dict()
            results.append({"score": float(score), **row})
        return results
