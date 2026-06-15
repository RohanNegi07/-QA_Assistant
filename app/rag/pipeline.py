import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from app.config import settings

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover - optional dependency path
    ChatGroq = None
    ChatPromptTemplate = None


class RAGPipeline:
    def __init__(self):
        self._df = None

    def initialize(self):
        data_path = Path("Answers.csv")
        if not data_path.exists():
            self._df = pd.DataFrame(columns=["Id", "Body", "Score"])
            return

        df = pd.read_csv(
            data_path,
            encoding="latin-1",
            usecols=["Id", "Score", "Body"],
            nrows=20000,
        )
        df = df.dropna(subset=["Body"])
        df = df[df["Score"] >= 3].copy()
        df["Body"] = df["Body"].astype(str)
        df["clean_text"] = df["Body"].apply(self._clean_html)
        self._df = df

    def _clean_html(self, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        for code in soup.find_all("code"):
            code.string = f"\n```\n{code.get_text()}\n```\n"
        return re.sub(r"\s+", " ", soup.get_text(" ")).strip()

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]{3,}", text.lower()))

    async def ask(self, question: str) -> dict:
        if self._df is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        if self._df.empty:
            return {
                "answer": "No Stack Overflow dataset file is available in the workspace yet.",
                "sources": [],
            }

        question_tokens = self._tokenize(question)
        scored = []
        for _, row in self._df.iterrows():
            tokens = self._tokenize(row["clean_text"])
            overlap = len(question_tokens & tokens)
            score = overlap + 0.05 * min(len(question_tokens), 10)
            if overlap > 0:
                scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [item[1] for item in scored[: settings.top_k]]

        if not top_docs:
            return {
                "answer": "No strong matches were found in the local Stack Overflow dataset for that question.",
                "sources": [],
            }

        context = "\n\n".join(doc["clean_text"][:800] for doc in top_docs[:3])
        sources = [
            {
                "title": f"Stack Overflow answer #{int(doc['Id'])}",
                "score": int(doc.get("Score", 0)),
                "snippet": doc["clean_text"][:180],
            }
            for doc in top_docs[:3]
        ]

        if settings.groq_api_key and ChatGroq is not None and ChatPromptTemplate is not None:
            try:
                llm = ChatGroq(
                    model=settings.groq_model,
                    api_key=settings.groq_api_key,
                    temperature=0.2,
                )
                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a careful Python programming assistant. Use the provided Stack Overflow context to answer the user. If the context is insufficient, say so clearly and provide practical general advice.",
                        ),
                        ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer with a concise explanation and a small code example when helpful."),
                    ]
                )
                chain = prompt | llm
                response = await chain.ainvoke({"context": context, "question": question})
                answer = getattr(response, "content", str(response)).strip()
                if answer:
                    return {"answer": answer, "sources": sources}
            except Exception:
                pass

        answer = "\n\n".join(doc["clean_text"][:500] for doc in top_docs[:3])
        return {"answer": answer, "sources": sources}


rag_pipeline = RAGPipeline()
