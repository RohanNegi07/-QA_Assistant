import json
from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = Path("data/sample_qa.jsonl")
CHROMA_PATH = Path("./chroma_db")


def load_documents(path: Path):
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            content = (
                f"QUESTION: {item['title']}\n\n"
                f"{item['question']}\n\n"
                f"ANSWER: {item['answer']}"
            )
            docs.append(
                Document(
                    page_content=content,
                    metadata={"id": item["id"], "title": item["title"]},
                )
            )
    return docs


def build_index():
    docs = load_documents(DATA_PATH)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    CHROMA_PATH.mkdir(exist_ok=True)
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH),
    )
    print(f"Built sample index with {len(docs)} documents at {CHROMA_PATH}")


if __name__ == "__main__":
    build_index()
