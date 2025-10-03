from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"

def build_index(chunks_parquet: Path, index_path: Path, embedding_model_name=MODEL_NAME):
    df = pd.read_parquet(chunks_parquet)
    texts = df["text"].tolist()
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    df.reset_index(drop=True, inplace=True)
    df["chunk_id"] = df.index
    df.to_parquet(chunks_parquet, index=False)
    return index, df

def load_index(index_path: Path):
    return faiss.read_index(str(index_path))
