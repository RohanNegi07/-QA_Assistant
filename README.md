# Python Programming Q&A Assistant

This project is a FastAPI-based Python Q&A assistant that answers Python questions using the local Stack Overflow answer dataset in this workspace.

## What is included

- FastAPI backend with GET /health and POST /ask
- Local retrieval pipeline over the provided Answers.csv dataset
- Basic pytest coverage for API behavior
- Test query notes for assessment-style validation

## Quick start

1. Create a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Start the API: `uvicorn app.main:app --reload`
4. Run tests: `pytest tests/ -v`

## API endpoints

- GET /health
  - Returns service status and readiness flag
- POST /ask
  - Body: `{"question": "How do I reverse a list in Python?"}`
  - Returns `answer`, `sources`, and `model`

## Current implementation notes

- The current retrieval path uses the local dataset file `Answers.csv` and returns the most relevant answer snippets from the workspace.
- No external LLM key is required for this local version.
- If you want the full cloud/LLM-enhanced variant, add `GROQ_API_KEY` or `ANTHROPIC_API_KEY` in `.env` and extend the retrieval pipeline.

## Verification

Run the test suite with:

`pytest tests/ -v`

Current verified result: 4 tests passed.

## Deployment

This repo includes a container-friendly deployment setup:

- `Dockerfile` for standard container hosting
- `render.yaml` for Render deployment

To deploy:

1. Set `GROQ_API_KEY` (and/or `ANTHROPIC_API_KEY`) in your hosting platform environment variables.
2. Build from the repo root with `docker build -t python-qa-assistant .` or use the Render config.
3. Start with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

A public live URL can be added once the app is hosted on your chosen platform.
