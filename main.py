"""
main.py - CLI Interface for the PDF RAG System

Provides a simple command-line loop so users can:
  1. Supply a PDF file path as a command-line argument.
  2. Ask multiple questions interactively.
  3. Type 'exit' or 'quit' to stop.

The heavy PDF processing (extraction, chunking, embedding, indexing) is done
ONCE at startup. Each subsequent question only requires an embedding + FAISS
search + Gemini call, making the loop fast after the first load.
"""

import sys
from rag import (
    extract_text,
    chunk_text,
    create_embeddings,
    build_faiss_index,
    retrieve_chunks,
    generate_answer,
)


def main():
    # ------------------------------------------------------------------
    # Parse CLI argument: path to PDF
    # ------------------------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage:  python main.py <path_to_pdf>")
        print("Example: python main.py research_paper.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print("=" * 60)
    print("       PDF Question Answering System  (RAG from scratch)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # One-time pipeline setup
    # ------------------------------------------------------------------
    try:
        # Step 1: Extract raw text from the PDF
        full_text = extract_text(pdf_path)

        # Step 2: Chunk the text with overlap
        chunks = chunk_text(full_text)

        # Step 3: Generate embeddings for all chunks
        embeddings = create_embeddings(chunks)

        # Step 4: Build the FAISS index from embeddings
        index = build_faiss_index(embeddings)

    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Unexpected Error during setup] {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Setup complete! Ask your questions below.")
    print("Type  'exit'  or  'quit'  to stop.\n")

    # ------------------------------------------------------------------
    # Interactive Q&A loop
    # ------------------------------------------------------------------
    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+C or piped input gracefully
            print("\n\nExiting. Goodbye!")
            break

        if not question:
            print("  (Please enter a question.)\n")
            continue

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        print()  # blank line for readability

        try:
            # Steps 5-6: Retrieve relevant chunks from FAISS
            relevant_chunks = retrieve_chunks(question, chunks, index)

            # Steps 7-9: Generate answer via Gemini
            answer = generate_answer(question, relevant_chunks)

        except EnvironmentError as e:
            print(f"[Config Error] {e}\n")
            continue
        except Exception as e:
            print(f"[Error while answering] {e}\n")
            continue

        print("\n" + "-" * 60)
        print("ANSWER:")
        print(answer)
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
