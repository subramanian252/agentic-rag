# Agentic RAG

A portfolio demonstration built from `notebooks/agentic_rag.ipynb`. It keeps the notebook's agentic routing between a vector retriever, Tavily web search, and direct generation behind one FastAPI application rendered with Jinja templates.

## Source and build provenance

The RAG workflow, prompts, graph stages, routing labels, grading behavior, and retry limits come from the included notebook. The FastAPI/Jinja infrastructure, repository structure, browser interface, and this documentation were built by AI from that notebook source.

The current local knowledge source is the machine-learning PDF collection in `backend/data`, not a web-page loader. Retrieval uses the same saved Pinecone index, `agenticrag`, as Corrective RAG and Self-RAG. Tavily remains the secondary source for recent or external information.

## Workflow

The graph selects one of three routes:

- `retriever` for questions covered by the saved knowledge base.
- `websearch` for recent, changing, or external information.
- `generate_answer` for general questions that require neither source.

The retriever route grades Pinecone evidence before generation. Weak knowledge-base evidence moves to Tavily. Web results are also graded; insufficient results trigger query rewriting and another search, up to the graph's three-loop limit.

```text
Question -> Route
            | general -> Direct answer -> End
            | knowledge base -> Pinecone -> Grade -> Answer
            |                                  ` weak -> Web search
            ` current/external -> Web search -> Grade
                                             | enough -> Answer
                                             ` weak -> Rewrite -> Search
```

## Shared Pinecone index

All three portfolio applications connect to `agenticrag`.

- If the index does not exist, the application creates a 1536-dimension cosine index in AWS `us-east-1`, loads every `backend/data/*.pdf` file, chunks the documents with a size of 900 and overlap of 100, and uploads the embeddings.
- If the index already exists, it opens the saved vectors directly and skips PDF ingestion.

The expected workflow is to let Corrective RAG create and populate the index once, then reuse the saved index in this application.

## API

FastAPI renders the page at `GET /` and accepts questions at `POST /query`.

```http
POST /query
Content-Type: application/json

{"question": "What is machine learning?"}
```

```json
{"answer": "..."}
```

## Project structure

```text
agentic-rag/
|-- app/
|   |-- __init__.py
|   `-- main.py
|-- backend/
|   |-- app/
|   |   |-- __init__.py
|   |   `-- graph.py
|   `-- data/                 # Machine-learning PDF collection
|-- frontend/
|   |-- index.html
|   |-- styles.css
|   `-- app.js
|-- notebooks/
|   `-- agentic_rag.ipynb
|-- .env
|-- .gitignore
|-- .python-version
|-- requirements.txt
|-- vercel.json
`-- README.md
```

## Requirements

- Python 3.12
- OpenRouter API key
- Pinecone API key
- Tavily API key

The project-root `.env` must provide:

```dotenv
OPENROUTER_API_KEY=...
PINECONE_DB=...
TAVILY_API_KEY=...
```

The `.env` file is ignored by Git and must not be committed.

## Run locally

From the `agentic-rag` folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:api --reload
```

Open `http://127.0.0.1:8000`.

The graph initializes on the first question. When `agenticrag` already exists, the application connects to its saved vectors without loading the local PDFs again.

## Deploy to Vercel

This repository is structured for Vercel's FastAPI runtime:

- `app/main.py` exports the FastAPI application as both `api` and `app`.
- `requirements.txt` is at the repository root for dependency detection.
- `.python-version` selects Python 3.12.
- `vercel.json` allows the FastAPI function to run for up to 300 seconds.

Create and populate the shared `agenticrag` Pinecone index with Corrective RAG before deploying. Import this repository as its own Vercel project, leave the build and output-directory settings empty, and configure `OPENROUTER_API_KEY`, `PINECONE_DB`, and `TAVILY_API_KEY` for Preview and Production. Do not upload or commit the local `.env` file.

## Notebook-to-application mapping

| Notebook concept | Application location |
| --- | --- |
| Directory PDF loading | `backend/app/graph.py` module setup |
| Pinecone index and `k=6` retriever | Module setup |
| Three-way route selection | `decider_node` |
| Knowledge-base grading | `gradekb_node` |
| Tavily search and grading | `websearch_node` and `grade_web_answer_node` |
| Three-loop rewrite fallback | `rewrite_query_node` and `grade_web_router` |
| Final answer paths | Corresponding graph nodes |
| Graph invocation | FastAPI `POST /query` |

## Troubleshooting

- Pinecone authentication or index errors: verify `PINECONE_DB`, account permissions, region availability, and index dimension.
- PDF loading errors during initial creation: confirm the files exist under `backend/data` and reinstall `pypdf`.
- OpenRouter errors: verify `OPENROUTER_API_KEY` and model access.
- Tavily errors: verify `TAVILY_API_KEY`.
- Browser network errors: confirm Uvicorn is running and open `http://127.0.0.1:8000`.
