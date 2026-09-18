from pathlib import Path
from sentence_transformers import SentenceTransformer
from chunking import load_extracted_text, create_chunks


if __name__ == "__main__":

    # Load extracted text
    text_path = Path("data/extracted_text.txt")
    text = load_extracted_text(text_path)

    # Create chunks
    chunks = create_chunks(text)

    print("================================")
    print("CREATING EMBEDDINGS")
    print("================================")
    print(f"Total chunks: {len(chunks)}")

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert chunks into vectors
    embeddings = model.encode(
        chunks,
        show_progress_bar=True
    )

    print("\n================================")
    print("EMBEDDINGS CREATED")
    print("================================")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding size: {embeddings.shape[1]}")