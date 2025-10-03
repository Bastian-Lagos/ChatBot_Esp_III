import argparse
import os
import asyncio
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from rag.retrieve import Retriever
import pandas as pd
import faiss
from rag.prompts import SYSTEM_PROMPT, USER_PROMPT_WITH_CONTEXT
from dotenv import load_dotenv

load_dotenv()

def format_context(retrieved):
    lines = []
    for r in retrieved:
        title = r.get("title", r.get("doc_id"))
        page = r.get("page", "?")
        text = r.get("text", "")[:800].replace("\n", " ")
        lines.append(f"[{title}, página {page}]: {text}")
    return "\n\n".join(lines)

def interactive(provider_name="chatgpt", k=3):
    index = faiss.read_index(os.getenv("FAISS_INDEX_PATH", "data/index.faiss"))
    chunks = pd.read_parquet(os.getenv("CHUNKS_PARQUET", "data/processed/chunks.parquet"))
    retriever = Retriever(index, chunks)
    provider = ChatGPTProvider() if provider_name=="chatgpt" else DeepSeekProvider()
    print(f"Usando proveedor: {provider.name}")
    while True:
        q = input("\nPregunta (enter para salir): ").strip()
        if not q:
            break
        retrieved = retriever.retrieve(q, k=k)
        context = format_context(retrieved)
        system = {"role":"system","content":SYSTEM_PROMPT}
        user_msg = USER_PROMPT_WITH_CONTEXT.format(question=q, context=context)
        messages = [system, {"role":"user","content": user_msg}]
        resp = provider.chat(messages, max_tokens=512)
        print("\n--- RESPUESTA ---\n")
        print(resp["text"])
        print("\n--- CITAS ---")
        for r in retrieved:
            print(f"{r['tittle']} - página {r['page']} (score {r['score']:.3f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["chatgpt","deepseek"], default="chatgpt")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    interactive(provider_name=args.provider, k=args.k)
