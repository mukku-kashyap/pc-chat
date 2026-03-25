from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from pc_rag_ingestion import sync_data
from pc_rag_retrieval import get_agent
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # This allows OPTIONS, POST, GET, etc.
    allow_headers=["*"],
)

agent = None

class Query(BaseModel):
    question: str
    session_id: str = None


@app.on_event("startup")
def load_rag():
    global agent
    print("🚀 Loading RAG system...")

    page_index = sync_data(reset=False)
    agent = get_agent(
        llm=ChatGroq(model="llama-3.1-8b-instant"),
        page_index=page_index,
        domain="@alliedbenefit.com",
        key="allied"
    )
    print("✅ RAG Loaded")


@app.post("/ask")
def ask(query: Query):
    global agent
    session_id = query.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    result = agent.graph.invoke(
        {"input": query.question},
        config=config
    )
    return {
        "answer": result["answer"],
        "session_id": session_id
    }
