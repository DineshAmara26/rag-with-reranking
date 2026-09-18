from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from chunking import load_extracted_text, create_chunks


# Step 1: Load extracted text
text_path = Path("data/extracted_text.txt")
text = load_extracted_text(text_path)

# Step 2: Create chunks
chunks = create_chunks(text)

print(f"Total chunks: {len(chunks)}")

# Step 3: Convert chunks into LangChain Documents
documents = [Document(page_content=chunk) for chunk in chunks]

# Step 4: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Wrapper for SentenceTransformer
class SentenceTransformerEmbeddings:
    def embed_documents(self, texts):
        return model.encode(texts).tolist()

    def embed_query(self, text):
        return model.encode(text).tolist()

embedding_function = SentenceTransformerEmbeddings()

# Step 5: Create FAISS vector database
db = FAISS.from_documents(documents, embedding_function)

# Step 6: Save database
db.save_local("faiss_index")

print("\n===============================")
print("FAISS DATABASE CREATED")
print("===============================")
print("Saved inside folder: faiss_index")