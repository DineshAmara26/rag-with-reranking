from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_extracted_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    return chunks


if __name__ == "__main__":
    text_path = Path("data/extracted_text.txt")

    text = load_extracted_text(text_path)

    chunks = create_chunks(text)

    print("================================")
    print("CHUNKING COMPLETED")
    print("================================")
    print(f"Total chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk[:500])