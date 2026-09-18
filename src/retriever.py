from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Load the same embedding model used to create the FAISS database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load the existing FAISS database
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


def search_documents(query, k=3):
    results = db.similarity_search_with_score(query, k=k)
    return results


if __name__ == "__main__":

    question = input("\nAsk a question about the PDF: ")

    results = search_documents(question)

    print("\n================================")
    print("RETRIEVED DOCUMENTS")
    print("================================")

    for i, (document, score) in enumerate(results, start=1):

        print(f"\n--- Result {i} ---")
        print(f"Score: {score}")
        print(document.page_content[:1000])