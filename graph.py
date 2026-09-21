import os
from typing import List, TypedDict, Literal, Dict
from pydantic import BaseModel
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# -----------------------------
# Data + Index (Persistent FAISS)
# -----------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

import glob

if os.path.exists("faiss_index"):
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    for pdf_file in glob.glob("*.pdf"):
        docs = PyPDFLoader(pdf_file).load()
        chunks = text_splitter.split_documents(docs)
        for d in chunks:
            d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
            # add filename to metadata for citation purposes
            d.metadata["source"] = pdf_file
        all_chunks.extend(chunks)
        
    vector_store = FAISS.from_documents(all_chunks, embeddings)
    vector_store.save_local("faiss_index")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm = ChatOllama(model="llama3.1:8b", num_predict=1024)

UPPER_TH = 0.7
LOWER_TH = 0.3

# -----------------------------
# State
# -----------------------------
class State(TypedDict):
    question: str
    chat_history: List[Dict[str, str]]

    docs: List[Document]
    good_docs: List[Document]

    verdict: str
    reason: str

    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    web_query: str
    web_docs: List[Document]

    answer: str

# -----------------------------
# Retrieve
# -----------------------------
def retrieve_node(state: State) -> State:
    q = state["question"]
    return {"docs": retriever.invoke(q)}

# -----------------------------
# Score-based doc evaluator
# -----------------------------
class DocEvalScore(BaseModel):
    score: float
    reason: str

doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)

def eval_each_doc_node(state: State) -> State:
    q = state["question"]
    scores: List[float] = []
    good: List[Document] = []

    for d in state["docs"]:
        out = doc_eval_chain.invoke({"question": q, "chunk": d.page_content})
        scores.append(out.score)

        if out.score > LOWER_TH:
            good.append(d)

    if any(s > UPPER_TH for s in scores):
        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": f"At least one retrieved chunk scored > {UPPER_TH}.",
        }

    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All retrieved chunks scored < {LOWER_TH}.",
        }

    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}.",
    }

# -----------------------------
# Fast Semantic Filter
# -----------------------------
def refine(state: State) -> State:
    q = state["question"]

    if state.get("verdict") == "CORRECT":
        docs_to_use = state.get("good_docs", [])
    elif state.get("verdict") == "INCORRECT":
        docs_to_use = state.get("web_docs", [])
    else:
        docs_to_use = state.get("good_docs", []) + state.get("web_docs", [])

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()
    
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(context)
    
    kept: List[str] = []
    if chunks:
        # Ephemeral FAISS index for lighting-fast semantic filtering of raw web content
        temp_store = FAISS.from_texts(chunks, embeddings)
        rel_docs = temp_store.similarity_search(q, k=min(10, len(chunks)))
        kept = [d.page_content for d in rel_docs]

    refined_context = "\n".join(kept).strip()

    return {
        "strips": chunks,
        "kept_strips": kept,
        "refined_context": refined_context,
    }

# -----------------------------
# Query rewrite for web search
# -----------------------------
class WebQuery(BaseModel):
    query: str

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Consider the chat history if the user is asking a follow-up question.\n"
            "Rules:\n"
            "- Keep it short (6–14 words).\n"
            "- If the question implies recency, add a constraint like (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query",
        ),
        ("human", "Chat History:\n{chat_history}\n\nLatest Question: {question}"),
    ]
)

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)

def rewrite_query_node(state: State) -> State:
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.get("chat_history", [])])
    out = rewrite_chain.invoke({"question": state["question"], "chat_history": history_str})
    return {"web_query": out.query}

# -----------------------------
# Web search node
# -----------------------------
tavily = TavilySearchResults(max_results=5, include_raw_content=True)

def web_search_node(state: State) -> State:
    q = state.get("web_query") or state["question"]
    results = tavily.invoke({"query": q})

    web_docs: List[Document] = []
    for r in results or []:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("raw_content", "") or r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))

    return {"web_docs": web_docs}

# -----------------------------
# Generate
# -----------------------------
answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
            "If the context is empty or insufficient, say: 'I don't know.'\n"
            "Use the chat history for context if the user asks a follow-up.\n"
            "Context:\n{context}"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

def generate(state: State) -> State:
    messages = []
    for msg in state.get("chat_history", []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
            
    messages.append(HumanMessage(content=state["question"]))
    
    out = (answer_prompt | llm).invoke({"messages": messages, "context": state["refined_context"]})
    return {"answer": out.content}

# -----------------------------
# Routing
# -----------------------------
def route_after_eval(state: State) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"
    else:
        return "rewrite_query"

# -----------------------------
# Build graph
# -----------------------------
g = StateGraph(State)

g.add_node("retrieve", retrieve_node)
g.add_node("eval_each_doc", eval_each_doc_node)
g.add_node("rewrite_query", rewrite_query_node)
g.add_node("web_search", web_search_node)
g.add_node("refine", refine)
g.add_node("generate", generate)

g.add_edge(START, "retrieve")
g.add_edge("retrieve", "eval_each_doc")

g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {
        "refine": "refine",
        "rewrite_query": "rewrite_query",
    },
)
g.add_edge("rewrite_query", "web_search")
g.add_edge("web_search", "refine")
g.add_edge("refine", "generate")
g.add_edge("generate", END)

app = g.compile()
