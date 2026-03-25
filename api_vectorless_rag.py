from fastapi import FastAPI
from pydantic import BaseModel
from core_rag import rag_pipeline
import uuid
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all (for now)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str | None = None
    session_id: str | None = None

@app.post("/ask")
def ask(query: Query):
    if not query.question:
        return {"error": "Question is required"}
    session_id = query.session_id or str(uuid.uuid4())
    answer = rag_pipeline(query.question, session_id)
    return {
        "answer": answer,
        "session_id": session_id
    }
