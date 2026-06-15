from fastapi import APIRouter

from app.models import HealthResponse
from app.rag.pipeline import rag_pipeline

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        pipeline_ready=rag_pipeline._df is not None,
        version="1.0.0",
    )
