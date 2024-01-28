import logging
from typing import Tuple, List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from config import OPENAI_API_KEY

# Configuration
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context given here you will look at the remarks_text:
---
{context}
---

Answer the question based on the above context: {question}
Note: If you are unable to answer the question, please type "I don't know".
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def initialize_db(api_key: str, persist_directory: str) -> Chroma:
    """Initialize and return the Chroma database."""
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_function)


def search_db(db: Chroma, query_text: str, threshold: float = 0.7, k: int = 2) -> Optional[str]:
    """Search the database and return concatenated context if relevant documents are found."""
    results = db.similarity_search_with_relevance_scores(query_text, k=k)
    if len(results) == 0 or results[0][1] < threshold:
        logging.warning("Unable to find matching results with sufficient relevance.")
        return None

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    return context_text


def generate_prompt(context: str, question: str) -> str:
    """Generate a prompt using the given context and question."""
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt_template.format(context=context, question=question)


def get_response(api_key: str, prompt: str, temperature: float = 0.7, max_tokens: int = 100) -> str:
    """Generate a response from the model based on the prompt."""
    model = ChatOpenAI(
        openai_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return model.predict(prompt)


def format_response(response_text: str, sources: List[str]) -> str:
    """Format the response text with sources."""
    return f"Response: {response_text}\n\nSources: {sources}"


def main():
    prompt_input = input("Enter a prompt: ")
    db = initialize_db(OPENAI_API_KEY, CHROMA_PATH)
    context_text = search_db(db, prompt_input)

    if context_text:
        prompt = generate_prompt(context_text, prompt_input)
        response_text = get_response(OPENAI_API_KEY, prompt)
        # Assuming you modify your DB search to also return sources
        sources = [doc.metadata.get("source", None) for doc, _score in
                   db.similarity_search_with_relevance_scores(prompt_input, k=2)]
        formatted_response = format_response(response_text, sources)
        print(formatted_response)
    else:
        print("No relevant context found to generate a response.")


if __name__ == "__main__":
    main()
