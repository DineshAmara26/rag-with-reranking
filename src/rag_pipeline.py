from sentence_transformers import CrossEncoder
from retriever import search_documents


# Load the reranker model
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def retrieve_and_rerank(query, retrieve_k=5, final_k=3):
    # Step 1: Retrieve documents using FAISS
    retrieved_documents = search_documents(
        query,
        k=retrieve_k
    )

    # Step 2: Prepare query-document pairs
    pairs = [
        [query, document.page_content]
        for document, _ in retrieved_documents
    ]

    # Step 3: Calculate reranker scores
    scores = reranker.predict(pairs)

    # Step 4: Combine documents with scores
    ranked_results = list(
        zip(retrieved_documents, scores)
    )

    # Step 5: Sort by relevance
    ranked_results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    # Step 6: Return the best results
    return ranked_results[:final_k]


if __name__ == "__main__":

    question = input("\nAsk a question about the PDF: ")

    results = retrieve_and_rerank(question)

    print("\n================================")
    print("FINAL RERANKED RESULTS")
    print("================================")

    for i, ((document, faiss_score), reranker_score) in enumerate(
        results,
        start=1
    ):
        print(f"\n--- Result {i} ---")
        print(f"FAISS Score: {faiss_score:.4f}")
        print(f"Reranker Score: {reranker_score:.4f}")
        print("\nContent:")
        print(document.page_content[:800])