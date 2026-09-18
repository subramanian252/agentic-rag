# Agentic RAG

A portfolio application built from the original `agentic_rag.ipynb` notebook. The project retains the notebook's agentic routing between a Pinecone knowledge base, Tavily web search, and direct generation, and exposes that graph through one small FastAPI application.

The interface is visually aligned with the provided AI-engineer portfolio reference: an editorial paper canvas, dark green ink, a simple illustrated landscape, handwritten detail, and a restrained project-card layout. It intentionally provides only the notebook's core question-and-answer experience.

## Source and build provenance

All RAG behavior, prompts, graph steps, routing labels, retry limits, retrieval settings, and source information in this project are sourced from the included `notebooks/agentic_rag.ipynb` notebook. The application infrastructure—including the FastAPI/Jinja integration, project structure, browser interface, and documentation—was built by AI from that notebook source.

The Pinecone knowledge base is built from the official **LangGraph Agentic RAG documentation** loaded by the notebook. Tavily web search is the secondary source for recent or external questions; general questions can follow the notebook's direct-answer route.

## What it does

The graph begins by classifying the question into one of three routes:

- `retriever` for LangChain, LangGraph, custom RAG, retrieval, grading, and query-rewriting documentation.
- `websearch` for recent, changing, or external information.
- `generate_answer` for general questions that need neither source.

The retrieval route grades knowledge-base evidence before generation. If it is insufficient, the graph moves to web search. Web results are also graded; insufficient results trigger query rewriting and another search, up to three loops, before a limited-information fallback answer is returned.

## Architecture

```text
Browser → POST /query → FastAPI → Decider
                                  ├─ Direct answer ─────────────────────→ END
                                  ├─ Pinecone retrieve → Grade KB
                                  │                       ├─ enough → Generate → END
                                  │                       └─ weak ─┐
                                  └─ Web search ←──────────────────┘
                                            ↓
                                       Grade web
                                      ↙         ↘
                                  enough       rewrite query
                                    ↓              │
                                 Generate     search again
                                    ↓              │
                                   END        fallback after 3
```

FastAPI serves the Jinja page at `GET /` and accepts questions at `POST /query`:

```http
POST /query
Content-Type: application/json

{"question": "What is custom RAG?"}
```

Response:

```json
{"answer": "..."}
```

## Project structure

```text
agentic-rag/
├── app/
│   ├── __init__.py
│   └── main.py            # FastAPI + Jinja application entry point
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── graph.py       # Agent nodes and LangGraph wiring
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── notebooks/
│   └── agentic_rag.ipynb  # Unmodified source notebook
├── .env                   # Copied locally; ignored by Git
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or newer
- OpenRouter API key
- Pinecone API key
- Tavily API key
- Network access to load the LangGraph documentation used by the notebook

The copied project-root `.env` must provide the variable names used by the notebook:

```dotenv
OPENROUTER_API_KEY=...
PINECONE_DB=...
TAVILY_API_KEY=...
```

Never commit `.env`; it is included in `.gitignore`.

## Run locally

From the project root, run one application process:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn app.main:api --reload
```

Open `http://127.0.0.1:8000`. The same FastAPI process renders the Jinja frontend, serves its CSS and JavaScript, and handles `/query`.

The web page starts immediately. On the first question, the application follows the notebook setup:

1. Load the LangGraph Agentic RAG documentation page.
2. Split it into 900-character chunks with 100-character overlap.
3. Connect to the `agenticrag` Pinecone index, creating a 1536-dimension cosine index in AWS `us-east-1` if absent.
4. Add the documentation chunks.
5. Build the `k=6` retriever and compile the graph.

Because setup accesses external services, backend startup requires valid credentials and may take longer than the other two projects.

## Notebook-to-app mapping

| Notebook element | Application location |
| --- | --- |
| LangGraph documentation loader | `backend/app/graph.py` module setup |
| Pinecone index and `k=6` retriever | module setup |
| Three-way route selection | `decider_node` |
| Knowledge-base grading | `gradekb_node` |
| Tavily search and grading | `websearch_node` and `grade_web_answer_node` |
| Three-loop rewrite fallback | `rewrite_query_node` and `grade_web_router` |
| Direct, KB, web, and fallback answers | their corresponding graph nodes |
| Notebook `app.invoke(...)` | FastAPI `POST /query` |

## Intentional notebook adaptations

- The project-root `.env` is loaded through an absolute path so startup is independent of the current working directory.
- The rewritten query is passed to subsequent web searches; this makes the notebook's rewrite loop perform its intended work.
- The three possible final state keys are normalized into one `{ "answer": string }` API response.
- Routing labels, index configuration, model, retrieval count, loop limit, and source URL remain aligned with the notebook.

## Notes

- The notebook adds documentation chunks to Pinecone at setup time, and this application preserves that behavior.
- The code expects an embedding model compatible with the notebook's 1536-dimension Pinecone index.
- The frontend calls the same-origin `/query` endpoint, so no separate frontend server or CORS setup is needed.
- Current-information questions can require several search and grading calls.

## Troubleshooting

- Pinecone authentication or index errors: verify `PINECONE_DB`, account permissions, region availability, and index dimension.
- LangGraph documentation load errors: confirm outbound network access and retry startup.
- OpenRouter errors: confirm `OPENROUTER_API_KEY` and access to `openai/gpt-4o-mini` plus the embedding model.
- Tavily errors: confirm `TAVILY_API_KEY`.
- Browser network error: confirm Uvicorn is running at `127.0.0.1:8000` and serve the frontend from its own directory.
