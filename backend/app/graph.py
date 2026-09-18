"""Agentic RAG graph adapted directly from notebooks/agentic_rag.ipynb."""

import os
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

loader = WebBaseLoader(
    "https://docs.langchain.com/oss/python/langgraph/agentic-rag#build-a-custom-rag-agent-with-langgraph"
)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
pc = Pinecone(api_key=os.getenv("PINECONE_DB"))
if not pc.has_index("agenticrag"):
    pc.create_index(
        name="agenticrag",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index("agenticrag")
vector_store_pine = PineconeVectorStore(index=index, embedding=embeddings)
vector_store_pine.add_documents(chunks)
retriever = vector_store_pine.as_retriever(search_kwargs={"k": 6})

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
)
tavily_tool = TavilySearch(max_results=3, topic="general", search_depth="advanced")


class AgentState(TypedDict):
    question: str
    route: Literal["retriever", "websearch", "generate_answer"]
    context: list[Document]
    grade_kb: bool
    web_results: list[dict[str, Any]]
    grade_web: bool
    generated_kb_answer: str
    generated_web_answer: str
    loops: int
    general_answer: str
    direct_answer_or_fallback_answer: str
    rewritten_question: str


class Decider(BaseModel):
    route: Literal["retriever", "websearch", "generate_answer"]


def decider_node(state: AgentState):
    decision = llm.with_structured_output(Decider).invoke(
        [
            SystemMessage(
                content="""
You are a routing classifier.

Choose exactly one route:

"retriever" for the indexed LangChain, LangGraph, custom RAG, retrieval, grading, and query-rewriting documentation.
"websearch" for current, changing, recent, or external information.
"generate_answer" for general knowledge that requires neither source.

Do not answer the question. Only choose the route.
"""
            ),
            HumanMessage(content=state["question"]),
        ]
    )
    return {"route": decision.route}


def retriever_node(state: AgentState):
    query = state.get("rewritten_question", state["question"])
    return {"context": retriever.invoke(query)}


class GradeKB(BaseModel):
    relevant: bool


def gradekb_node(state: AgentState):
    numbered_docs = "\n".join(
        f"index {i}: {doc.page_content}" for i, doc in enumerate(state["context"])
    )
    relevant = llm.with_structured_output(GradeKB).invoke(
        [
            SystemMessage(
                content="Judge whether the retrieved context contains enough information "
                "to answer the question. Return only the relevance decision."
            ),
            HumanMessage(
                content=f"Context: {numbered_docs}\n\nQuery: {state['question']}"
            ),
        ]
    )
    return {"grade_kb": relevant.relevant}


def generate_kb_answer_node(state: AgentState):
    numbered_docs = "\n".join(
        f"index {i}: {doc.page_content}" for i, doc in enumerate(state["context"])
    )
    response = llm.invoke(
        [
            SystemMessage(
                content="Answer the question based on the provided internal context. Give only the answer."
            ),
            HumanMessage(
                content=f"Question: {state['question']}\n\nContext:\n{numbered_docs}"
            ),
        ]
    )
    return {"generated_kb_answer": response.content}


def websearch_node(state: AgentState):
    query = state.get("rewritten_question", state["question"])
    search = tavily_tool.invoke(query)
    return {"web_results": search["results"]}


def grade_web_answer_node(state: AgentState):
    results = "\n".join(
        f"title: {item['title']} - content: {item['content']} - url: {item['url']}"
        for item in state["web_results"]
    )
    relevant = llm.with_structured_output(GradeKB).invoke(
        [
            SystemMessage(
                content="Determine whether the web results contain enough reliable evidence "
                "to answer the question accurately. Return only the relevance decision."
            ),
            HumanMessage(content=f"Context: {results}\n\nQuery: {state['question']}"),
        ]
    )
    return {"grade_web": relevant.relevant}


def generate_web_answer_node(state: AgentState):
    results = "\n".join(
        f"title: {item['title']} - content: {item['content']} - url: {item['url']}"
        for item in state["web_results"]
    )
    response = llm.invoke(
        [
            SystemMessage(
                content="Answer the question based on the provided web results. Give only the answer."
            ),
            HumanMessage(
                content=f"Question: {state['question']}\n\nWeb Results:\n{results}"
            ),
        ]
    )
    return {"generated_web_answer": response.content}


class QueryRewrite(BaseModel):
    question: str


rewrite_query_llm = llm.with_structured_output(QueryRewrite)


def rewrite_query_node(state: AgentState):
    response = rewrite_query_llm.invoke(
        [
            SystemMessage(
                content="Rewrite the user's query to make it more specific and useful for web search."
            ),
            HumanMessage(content=state["question"]),
        ]
    )
    return {
        "rewritten_question": response.question,
        "loops": state.get("loops", 0) + 1,
        "grade_web": None,
    }


def fallback_answer_node(state: AgentState):
    answer = llm.invoke(
        [
            SystemMessage(
                content="The internal documents and web results were insufficient for a complete "
                "answer. Say that clearly, then provide a general answer based on limited information."
            ),
            HumanMessage(
                content=f"question: {state['question']}\n\nWeb results: {state['web_results']}"
            ),
        ]
    )
    return {"direct_answer_or_fallback_answer": answer.content}


def generate_answer_node(state: AgentState):
    answer = llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content=state["question"]),
        ]
    )
    return {"direct_answer_or_fallback_answer": answer.content}


def decider_router(state: AgentState):
    return state["route"]


def grade_kb_router(state: AgentState):
    return "generate_kb_answer" if state["grade_kb"] else "websearch"


def grade_web_router(state: AgentState):
    if state.get("loops", 0) >= 3:
        return "fallback_answer"
    return "generate_web_answer" if state["grade_web"] else "rewrite_query"


graph = StateGraph(AgentState)
graph.add_node("decider", decider_node)
graph.add_node("retriever", retriever_node)
graph.add_node("gradekb", gradekb_node)
graph.add_node("generate_kb_answer", generate_kb_answer_node)
graph.add_node("websearch", websearch_node)
graph.add_node("gradeweb", grade_web_answer_node)
graph.add_node("generate_web_answer", generate_web_answer_node)
graph.add_node("rewrite_query", rewrite_query_node)
graph.add_node("fallback_answer", fallback_answer_node)
graph.add_node("generate_answer", generate_answer_node)
graph.add_edge(START, "decider")
graph.add_conditional_edges(
    "decider",
    decider_router,
    {
        "retriever": "retriever",
        "generate_answer": "generate_answer",
        "websearch": "websearch",
    },
)
graph.add_edge("retriever", "gradekb")
graph.add_conditional_edges(
    "gradekb",
    grade_kb_router,
    {"generate_kb_answer": "generate_kb_answer", "websearch": "websearch"},
)
graph.add_edge("generate_kb_answer", END)
graph.add_edge("websearch", "gradeweb")
graph.add_conditional_edges(
    "gradeweb",
    grade_web_router,
    {
        "generate_web_answer": "generate_web_answer",
        "rewrite_query": "rewrite_query",
        "fallback_answer": "fallback_answer",
    },
)
graph.add_edge("rewrite_query", "websearch")
graph.add_edge("generate_web_answer", END)
graph.add_edge("fallback_answer", END)
graph.add_edge("generate_answer", END)

app = graph.compile()
