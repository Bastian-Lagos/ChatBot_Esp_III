# UFRO Normativa Assistant (Chatbot RAG)

**Proyecto Práctico 1 – Asistente CHATBOT sobre normativa y reglamentos de la UFRO**  

Este proyecto implementa un asistente conversacional que responde preguntas sobre normativa universitaria de la UFRO, utilizando un **flujo RAG (Retrieval-Augmented Generation)** con **FAISS**, **sentence-transformers**, y **proveedores LLM ChatGPT y DeepSeek**.

---

## Estructura del proyecto

Chatbot/
├─ app.py # CLI principal (selección de proveedor, k, modo batch)
├─ providers/
│ ├─ base.py # Protocolo Provider
│ ├─ chatgpt.py # Adapter ChatGPT
│ └─ deepseek.py # Adapter DeepSeek
├─ rag/
│ ├─ ingest.py # Extracción de PDFs + chunking + metadatos
│ ├─ embed.py # Embeddings + FAISS
│ ├─ retrieve.py # Búsqueda vectorial y re-rank opcional
│ └─ prompts.py # Plantillas de sistema y usuario
├─ eval/
│ ├─ gold_set.jsonl # 20 preguntas de prueba con respuesta esperada y referencias
│ └─ evaluate.py # Cálculo de métricas (EM, similitud, cobertura, Prec@k)
├─ data/
│ ├─ raw/ # PDFs/HTML originales
│ ├─ processed/ # Chunks.parquet
│ └─ index.faiss # Índice FAISS persistido
├─ scripts/
│ └─ batch_demo.sh # Script batch para evaluar gold set, sin terminar
├─ .env # Variables de entorno
├─ README.md # Este archivo
└─ requirements.txt # Dependencias del proyecto

## Requisitos

- Python 3.11+
- Pip o Pipenv/venv
- Claves de API:
  - `OPENAI_API_KEY` (ChatGPT)
  - `DEEPSEEK_API_KEY` (DeepSeek)
- Modelos de embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Dependencias principales:

openai
faiss-cpu
sentence-transformers
pypdf
pandas
flask

### Instalación rápida:
pip install -r requirements.txt

### Variables de entorno
Crea un archivo .env:

OPENAI_API_KEY=tu_api_key_chatgpt
DEEPSEEK_API_KEY=tu_api_key_deepseek

### CLI interactiva

python app.py --provider chatgpt
python app.py --provider deepseek

Se te pedirá ingresar una pregunta.

El asistente devolverá la respuesta con referencias a los documentos si es que los encuentra.

### Modo batch (evaluación gold set)

python app.py --provider chatgpt --batch eval/gold_set.jsonl
python app.py --provider deepseek --batch eval/gold_set.jsonl
Genera un CSV con columnas: question, provider, answer, references, em, coverage, latency.

### Flujo RAG
Reescritura opcional de consulta

Recuperación top-k de chunks desde FAISS

Re-rank opcional según similitud semántica

Síntesis con LLM (ChatGPT o DeepSeek)

Política de abstención: si no hay soporte en normativa, indica "No encontrado en normativa UFRO"

### Datos y trazabilidad
PDFs/HTML originales: data/raw/

Chunks vectorizados: data/processed/chunks.parquet

Índice FAISS: data/index.faiss

Metadatos: doc_id, title, página, URL, vigencia

### Desarrollo y pruebas
rag/ingest.py → carga y limpia PDFs, crea chunks

rag/embed.py → genera embeddings y construye FAISS

rag/retrieve.py → busca top-k chunks

providers/ → adapters de ChatGPT y DeepSeek

eval/evaluate.py → calcula métricas sobre gold set

### Consideraciones éticas
Todas las respuestas incluyen citas verificables y referencias a documentos oficiales.

Política de abstención para evitar alucinaciones.

Privacidad: no se almacena información de usuarios fuera del uso de evaluación batch.

Vigencia normativa: se indica fecha de documento para cada referencia.

### Referencias
FAISS – Búsqueda vectorial eficiente

Sentence-Transformers – Embeddings semánticos

OpenAI API – ChatGPT

DeepSeek API – API LLM compatible OpenAI

