from pathlib import Path
from typing import Any, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def load(file_path) -> List[Document]:
    """Load documents."""
    docs = []
    for p in Path(file_path).rglob("*"):
        if p.is_dir():
            continue
        with open(p, 'r') as f:
            text = f.read()
        metadata = {"source": str(p)}
        print(text, metadata)
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

if __name__ == "__main__":

    #load local data
    raw_documents = load("./data/")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    #split and form vectorstore
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
