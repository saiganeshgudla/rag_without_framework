# PDF Question Answering — RAG from Scratch

A minimal, framework-free implementation of **Retrieval-Augmented Generation (RAG)** that lets you ask natural-language questions about any PDF document.

Built **without** LangChain or LlamaIndex — every step of the pipeline is implemented explicitly so you can see exactly how RAG works under the hood.

---

## How RAG Works in This Project

```
PDF File
   │
   ▼
[1] extract_text()        — pdfplumber reads every page and returns raw text
   │
   ▼
[2] chunk_text()          — text is split into overlapping character windows
   │                         (chunk_size=500, overlap=50)
   ▼
[3] create_embeddings()   — each chunk → 384-dim vector via all-MiniLM-L6-v2
   │
   ▼
[4] build_faiss_index()   — embeddings are loaded into a FAISS IndexFlatL2
   │
   │  ← user types a question
   ▼
[5] retrieve_chunks()     — question is embedded, FAISS finds top-3 closest chunks
   │
   ▼
[6] generate_answer()     — chunks + question → Gemini prompt → final answer
```

### Why Each Step Matters

| Step | Why it exists |
|---|---|
| **Chunking** | LLMs have a context window limit; chunking lets us index arbitrarily large PDFs |
| **Overlap** | Prevents important sentences at chunk boundaries from being split and lost |
| **Embeddings** | Converts text into vectors that capture *semantic meaning*, not just keywords |
| **FAISS** | Efficiently finds the most semantically similar chunks at query time |
| **Top-K retrieval** | Sends only the most relevant context to the LLM, keeping the prompt concise |
| **Grounded prompt** | Instructs Gemini to answer strictly from context; avoids hallucination |

---

## Project Structure

```
rag_without_framework/
├── rag.py           # Core pipeline: all 6 modular RAG functions
├── main.py          # CLI interface (interactive Q&A loop)
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will automatically download the `all-MiniLM-L6-v2` model (~90 MB) on the first run.

### 2. Set Your Gemini API Key

Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey), then:

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY = "your-api-key-here"
```

**Linux / macOS:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 3. Run the Program

```bash
python main.py path/to/your/document.pdf
```

---

## Usage

```
============================================================
       PDF Question Answering System  (RAG from scratch)
============================================================
[Step 1] Extracted 24,301 characters from 12 page(s).
[Step 2] Created 54 chunks (chunk_size=500, overlap=50)
[Step 3] Loading embedding model 'all-MiniLM-L6-v2'...
[Step 3] Embeddings shape: (54, 384)
[Step 4] FAISS index built with 54 vectors (dim=384).

Setup complete! Ask your questions below.
Type  'exit'  or  'quit'  to stop.

Your question: What is the main contribution of this paper?

[Step 5-6] Embedding query and searching FAISS index (top_k=3)...
  Rank 1: chunk #7  (L2 distance = 0.4821)
  Rank 2: chunk #8  (L2 distance = 0.5103)
  Rank 3: chunk #3  (L2 distance = 0.5671)
[Step 7-9] Sending prompt to Gemini (gemini-1.5-flash)...

------------------------------------------------------------
ANSWER:
The main contribution is ...
------------------------------------------------------------

Your question: exit
Goodbye!
```

---

## Configuration

You can tune these constants at the top of `rag.py`:

| Constant | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `3` | Retrieved chunks per query |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model to use |

---

## Error Handling

| Situation | Behaviour |
|---|---|
| PDF file not found | Prints a clear error and exits |
| PDF has no extractable text (image-only) | Raises `ValueError` with explanation |
| `GEMINI_API_KEY` not set | Prints a config error and skips the query |
| Unexpected API or model error | Prints the error and continues the loop |

---

## Dependencies

| Library | Purpose |
|---|---|
| `pdfplumber` | PDF text extraction |
| `sentence-transformers` | Local embedding model |
| `faiss-cpu` | Vector similarity search |
| `google-generativeai` | Gemini LLM API |
| `numpy` | Numerical arrays for FAISS |
