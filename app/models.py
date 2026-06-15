from typing import List

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)


class SourceDocument(BaseModel):
    title: str
    score: int
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    model: str = "local-rag"


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    version: str
