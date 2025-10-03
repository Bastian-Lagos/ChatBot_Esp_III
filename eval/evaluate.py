import json
from pathlib import Path
import pandas as pd
from providers.chatgpt import ChatGPTProvider
import faiss
from rag.retrieve import Retriever
from rag.prompts import SYSTEM_PROMPT, USER_PROMPT_WITH_CONTEXT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def load_gold(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)

def evaluate(gold_path, provider_name="chatgpt", k=5):
    index = faiss.read_index("data/index.faiss")
    chunks = pd.read_parquet("data/processed/chunks.parquet")
    retriever = Retriever(index, chunks)
    provider = ChatGPTProvider() if provider_name=="chatgpt" else None
    results = []
    for item in tqdm(list(load_gold(gold_path))):
        q = item["question"]
        retrieved = retriever.retrieve(q, k=k)
        context = "\n\n".join([f"[{r['tittle']}, página {r['page']}]: {r['text']}" for r in retrieved])
        system = {"role":"system","content":SYSTEM_PROMPT}
        user_msg = USER_PROMPT_WITH_CONTEXT.format(question=q, context=context)
        messages = [system, {"role":"user","content": user_msg}]
        resp = provider.chat(messages)
        answer = resp["text"]
        em = 1 if item["expected"].strip().lower() in answer.lower() else 0
        a_vec = MODEL.encode([answer])
        e_vec = MODEL.encode([item["expected"]])
        sim = float(cosine_similarity(a_vec, e_vec)[0,0])
        cites_ok = ("[" in answer and "página" in answer) or ("No encontrado en normativa UFRO" in answer)
        results.append({"id": item["id"], "em": em, "sim": sim, "cites": cites_ok, "answer": answer})
    df = pd.DataFrame(results)
    print(df[["id","em","sim","cites"]].mean())
    df.to_csv("eval/results.csv", index=False)

if __name__ == "__main__":
    evaluate("eval/gold_set.jsonl")
