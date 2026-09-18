from sentence_transformers import CrossEncoder
from retriever import search_documents


# Load reranking model
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_documents(query, documents):
    pairs = [
        [query, document.page_content]
        for document, _ in documents
    ]

    scores = reranker.predict(pairs)

    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_documents


if __name__ == "__main__":

    question = input("\nAsk a question about the PDF: ")

    # First retrieve candidates from FAISS
    results = search_documents(question, k=5)

    print("\n================================")
    print("FAISS RETRIEVAL")
    print("================================")

    for i, (document, score) in enumerate(results, start=1):
        print(f"\nResult {i} | FAISS Score: {score}")
        print(document.page_content[:500])

    # Then rerank the candidates
    ranked_results = rerank_documents(question, results)

    print("\n================================")
    print("RERANKED RESULTS")
    print("================================")

    for i, ((document, _), score) in enumerate(
        ranked_results, start=1
    ):
        print(f"\nResult {i} | Reranker Score: {score:.4f}")
        print(document.page_content[:500])