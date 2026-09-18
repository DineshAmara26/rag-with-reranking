import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from rag_pipeline import retrieve_and_rerank


# Load API key
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found.")


# Create Gemini client
client = genai.Client(api_key=api_key)


def generate_answer(question):
    # Retrieve and rerank relevant chunks
    results = retrieve_and_rerank(
        question,
        retrieve_k=5,
        final_k=3
    )

    # Combine retrieved information
    context_parts = []

    for i, ((document, _), score) in enumerate(results, start=1):
        context_parts.append(
            f"[Source {i}]\n{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    # Prompt for Gemini
    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not present in the context, say:
"I could not find the answer in the document."

Keep the answer clear and easy to understand.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate answer
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


if __name__ == "__main__":

    question = input("\nAsk a question about the PDF: ")

    answer = generate_answer(question)

    print("\n================================")
    print("FINAL ANSWER")
    print("================================")
    print(answer)