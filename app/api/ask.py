from fastapi import APIRouter, HTTPException, Request

from app.models import AskRequest, AskResponse
from app.rag.pipeline import rag_pipeline

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: Request, body: AskRequest):
    try:
        result = await rag_pipeline.ask(body.question)
        return AskResponse(
            question=body.question,
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
