from pypdf import PdfReader
from pathlib import Path
import re
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import pdfplumber

def ingest_OCR_REG_pdf(file_path: str, output_dir: str = None) -> list[str]:
    file_path = Path(file_path)
    pages = []
    articles = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text is None:
                text = ""
            text = re.sub(r"\n", " ", text)
            text = re.sub(r"\(cid:\d+\)", "", text)
            text = re.sub(r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑüÜ.,;:()\"'¿?¡!°%\-–—\s]", " ", text)
            text = text.strip()
            if text.isspace():
                continue
            if text.startswith("Reglamento de Convivencia Universitaria "):
                text = text[len("Reglamento de Convivencia Universitaria "):].strip()
            elif text.startswith("lamento de Convivencia Universitaria "):
                text = text[len("lamento de Convivencia Universitaria "):].strip()
            text = re.sub(r" Página \d+-", "", text)
            text = re.sub(r" -Página \d+", "", text)

            pages.append({"doc_id": file_path.stem, "page": i, "text": text})
    pages = pages[2:52]
    print(pages[0:2])
    return pages

def ingest_text_from_pdf(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
        text = re.sub(r"\s{2,}", " ", text)
        pages.append({"doc_id": path.stem, "page": i, "text": text})
    return pages[4:]

def classify_title(titles: list[str]) -> str:
    has_disposiciones = False
    has_articulos = False
    for title in titles:
        uptitle = title.upper()
        if "DISPOSICIONES" in uptitle:
            has_disposiciones = True
        if "ARTICULO" in uptitle or "ARTÍCULO" in uptitle:
            has_articulos = True
    if has_disposiciones and has_articulos:
        return "Disposiciones y articulos"
    elif has_disposiciones:
        return "Disposiciones"
    elif has_articulos:
        return "Articulos"
    else:
        return "Generales"

def chunk_texts(pages: list[dict], chunk_tokens=900, overlap_tokens=120):
    chunks = []
    for p in pages:
        words = p["text"].split()
        tokens_per_word = 1.0
        approx_words_per_chunk = int(chunk_tokens / tokens_per_word)
        step = approx_words_per_chunk - overlap_tokens
        if step <= 0:
            step = approx_words_per_chunk
        for i in range(0, max(1, len(words)), step):
            segment = " ".join(words[i:i+approx_words_per_chunk])
            if not segment.strip():
                continue
            tittles = re.sub(r'[^A-ZÁÉÍÓÚÑÜ]', ' ', segment).split()
            tittle_words = [w for w in tittles if len(w) > 1]
            final_tittle = classify_title(tittle_words)
            chunks.append({
                "doc_id": p["doc_id"],
                "page": p["page"],
                "text": segment,
                "tittle": final_tittle
            })
    return chunks

def ingest_text(filepath):
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        isChunk = False
        temp = ""
        for line in f:
            content = line.strip()
            if content == "" or content.isspace():
                continue
            if content.startswith("## Chunk") and not isChunk:
                isChunk = not isChunk
            elif content.startswith("## Chunk") and isChunk:
                chunks.append({"doc_id": Path(filepath).stem, "page": 1, "text": temp})
                temp = ""
            else:
                temp += " " + content
    return chunks

def create_faiss(chunks):
    all_chunks = []
    for sublist in chunks:
        for item in sublist:
            if isinstance(item, dict) and "text" in item:
                all_chunks.append(item)
            elif isinstance(item, str):
                all_chunks.append({"doc_id": "unknown", "page": -1, "text": item, "title": "general"})
            else:
                continue
    model = SentenceTransformer('all-MiniLM-L6-v2')
    rawTexts = [c["text"] for c in all_chunks]
    embeddings = model.encode(rawTexts, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, "data/index.faiss")

    df_chunks = pd.DataFrame(all_chunks)
    df_chunks.to_parquet("data/processed/chunks.parquet", index=False)

# Codigo para generar el indice FAISS
#testPages = ingest_OCR_REG_pdf("data/raw/OCR-REGLAMENTO-DE-CONVIVENCIA-UNIVERSITARIA.pdf")
#tokens = chunk_texts(testPages)
#regimenPages = ingest_text_from_pdf(Path("data/raw/ReglamentodeRegimendeEstudios__2023_.pdf"))
#regimenTokens = chunk_texts(regimenPages)
#calendarioPages = ingest_text("data/raw/calendario academico 2025.txt")
#create_faiss([tokens, regimenPages, calendarioPages])