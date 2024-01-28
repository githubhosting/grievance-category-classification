import os
import shutil
import json
from pprint import pprint

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from config import OPENAI_API_KEY

CHROMA_PATH = "chroma"


def load_csv_documents():
    file_path = "sample/sample_dataset.csv"
    loader = CSVLoader(file_path, encoding="utf-8", metadata_columns=["CategoryV7"])
    docs = loader.load()
    return docs


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == '__main__':
    documents = load_csv_documents()
    pprint(documents)
    save_to_chroma(documents)
