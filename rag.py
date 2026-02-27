"""
rag.py - Core RAG (Retrieval-Augmented Generation) Pipeline

This module implements a complete RAG pipeline from scratch:
  1. PDF text extraction
  2. Text chunking with overlap
  3. Embedding generation (sentence-transformers)
  4. FAISS vector index construction
  5. Semantic similarity search
  6. LLM-based answer generation (Google Gemini)

No LangChain or LlamaIndex is used.
"""

import os
import sys
import numpy as np
import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load GEMINI_API_KEY (and any other vars) from the .env file
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Embedding model - lightweight, fast, and very capable for RAG tasks
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunk settings
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 50      # characters of overlap between consecutive chunks

# Retrieval settings
TOP_K = 3               # number of top similar chunks to retrieve

# Google Gemini model
GEMINI_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Step 1: Extract text from PDF
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF file using pdfplumber.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A single string containing all extracted text from every page.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If no text could be extracted from the PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: '{pdf_path}'")

    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                all_text.append(text)
            else:
                print(f"  [Warning] Page {page_num} has no extractable text (may be an image-only page).")

    if not all_text:
        raise ValueError(
            "No text could be extracted from the PDF. "
            "The file may consist entirely of scanned images."
        )

    full_text = "\n".join(all_text)
    print(f"[Step 1] Extracted {len(full_text):,} characters from {len(all_text)} page(s).")
    return full_text


# ---------------------------------------------------------------------------
# Step 2: Split text into overlapping chunks
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long text into smaller, overlapping character-based chunks.

    Overlapping ensures that context near chunk boundaries is not lost.
    For example, with chunk_size=500 and overlap=50:
        chunk[0] covers chars   0 – 499
        chunk[1] covers chars 450 – 949
        chunk[2] covers chars 900 – 1399
        ...

    Args:
        text:       The full extracted text.
        chunk_size: Maximum number of characters in each chunk.
        overlap:    Number of characters shared between adjacent chunks.

    Returns:
        A list of text chunk strings.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        # Move the window forward by (chunk_size - overlap) to create overlap
        start += chunk_size - overlap

    # Remove any empty strings that may result from stripping
    chunks = [c for c in chunks if c]

    print(f"[Step 2] Created {len(chunks)} chunks "
          f"(chunk_size={chunk_size}, overlap={overlap}).")
    return chunks


# ---------------------------------------------------------------------------
# Step 3: Generate embeddings for each chunk
# ---------------------------------------------------------------------------

def create_embeddings(chunks: list[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """
    Convert a list of text chunks into dense vector embeddings.

    Uses the sentence-transformers library with the 'all-MiniLM-L6-v2' model,
    which produces 384-dimensional embeddings and is optimized for semantic
    similarity tasks.

    Args:
        chunks:     List of text chunks to embed.
        model_name: HuggingFace sentence-transformers model identifier.

    Returns:
        A NumPy array of shape (num_chunks, embedding_dim) with float32 dtype.
    """
    print(f"[Step 3] Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)

    print(f"[Step 3] Encoding {len(chunks)} chunks into embeddings...")
    # show_progress_bar provides friendly feedback during encoding
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    # FAISS requires float32
    embeddings = embeddings.astype(np.float32)

    print(f"[Step 3] Embeddings shape: {embeddings.shape}  "
          f"(chunks × embedding_dim)")
    return embeddings


# ---------------------------------------------------------------------------
# Step 4: Build a FAISS vector index
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a FAISS flat L2 index from the chunk embeddings.

    IndexFlatL2 performs exact nearest-neighbour search using Euclidean
    (L2) distance — the simplest and most accurate FAISS index type.
    For larger corpora you could switch to IndexIVFFlat or HNSW for speed.

    Args:
        embeddings: Float32 NumPy array of shape (num_chunks, embedding_dim).

    Returns:
        A trained and populated FAISS index ready for similarity search.
    """
    embedding_dim = embeddings.shape[1]

    # Create a flat (brute-force) L2 index
    index = faiss.IndexFlatL2(embedding_dim)

    # Add all chunk embeddings to the index
    index.add(embeddings)

    print(f"[Step 4] FAISS index built with {index.ntotal} vectors "
          f"(dim={embedding_dim}).")
    return index


# ---------------------------------------------------------------------------
# Step 5 & 6: Retrieve the top-k most relevant chunks
# ---------------------------------------------------------------------------

def retrieve_chunks(
    query: str,
    chunks: list[str],
    index: faiss.IndexFlatL2,
    model_name: str = EMBEDDING_MODEL_NAME,
    top_k: int = TOP_K,
) -> list[str]:
    """
    Embed the user query and find the top-k most similar chunks via FAISS.

    Process:
      1. Encode the query into a 384-dim embedding using the same model.
      2. Run FAISS .search() to find the k nearest neighbours (by L2 distance).
      3. Return the corresponding text chunks.

    Args:
        query:      The user's natural language question.
        chunks:     The original list of text chunks.
        index:      The populated FAISS index.
        model_name: The embedding model (must match what was used in Step 3).
        top_k:      Number of chunks to retrieve.

    Returns:
        A list of the top_k most relevant text chunks.
    """
    print(f"[Step 5-6] Embedding query and searching FAISS index (top_k={top_k})...")
    model = SentenceTransformer(model_name)

    # Encode the query; shape → (1, embedding_dim)
    query_embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)

    # Search: returns distances and indices of nearest neighbours
    distances, indices = index.search(query_embedding, top_k)

    retrieved = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        print(f"  Rank {rank}: chunk #{idx}  (L2 distance = {dist:.4f})")
        retrieved.append(chunks[idx])

    return retrieved


# ---------------------------------------------------------------------------
# Step 7 & 8 & 9: Generate an answer using Google Gemini
# ---------------------------------------------------------------------------

def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Build a prompt from retrieved context and send it to Google Gemini.

    RAG prompt strategy:
      - Provide only retrieved chunks as the knowledge source.
      - Instruct the model to answer STRICTLY from the provided context.
      - If the answer cannot be found, return a canned fallback response.

    Args:
        question:       The user's question.
        context_chunks: Top-k relevant text chunks from the document.

    Returns:
        The model's answer as a plain string.

    Raises:
        EnvironmentError: If GEMINI_API_KEY is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please export it before running the program."
        )

    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel(GEMINI_MODEL)

    # Build context block from retrieved chunks
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    # Carefully crafted prompt to stay grounded in the document
    prompt = f"""You are a precise document assistant. Answer the question using ONLY the context provided below.
If the answer is not present in the context, respond exactly with:
"Answer not found in the document."

Do not use any prior knowledge outside of the given context.

=== CONTEXT ===
{context}

=== QUESTION ===
{question}

=== ANSWER ==="""

    print(f"[Step 7-9] Sending prompt to Gemini ({GEMINI_MODEL})...")
    response = gemini.generate_content(prompt)

    return response.text.strip()
