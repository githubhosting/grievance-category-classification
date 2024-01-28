from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma

from config import OPENAI_API_KEY

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context given here you will look at the remarks_text:
---
{context}
---

Answer the question based on the above context: {question}
Note: If you are unable to answer the question, please type "I don't know".
"""


def get_response(prompt, results):
    model = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0.7,
        max_tokens=100
    )
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text.content}\n\nSources: {sources}"

    return formatted_response


def print_formatted_results(results):
    for i, (doc, similarity_score) in enumerate(results, start=1):
        page_content = doc.page_content
        category = doc.metadata.get('CategoryV7', 'N/A')
        row = doc.metadata.get('row', 'N/A')
        source = doc.metadata.get('source', 'N/A')

        formatted_result = f"Document {i}:\n" \
                           f"    Metadata:\n" \
                           f"        CategoryV7: '{category}'\n" \
                           f"        Row: {row}\n" \
                           f"        Source: '{source}'\n" \
                           f"    Similarity Score: {similarity_score}\n"
        print(formatted_result)


def main():
    prompt_input = input("Enter a prompt: ")
    query_text = prompt_input

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=2)
    print_formatted_results(results)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # print(f"Context: \n\n{context_text}\n\n")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # print(f"\nThe prompt: {prompt}\n\n")
    print(f"\n\n..................................\n\n")

    # Get response
    formatted_response = get_response(prompt, results)
    print(formatted_response)


if __name__ == "__main__":
    main()
