from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import ask, health
from app.rag.pipeline import rag_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing local retrieval pipeline...")
    rag_pipeline.initialize()
    print("Pipeline ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Python Q&A Assistant",
    description="RAG-powered Python programming Q&A assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/", tags=["root"])
async def root():
    return JSONResponse(
        status_code=200,
        content={"message": "Python Q&A Assistant API is running", "docs": "/docs"},
    )


app.include_router(health.router, tags=["health"])
app.include_router(ask.router, tags=["ask"])
