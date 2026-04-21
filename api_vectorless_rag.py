from fastapi import FastAPI, Form, Response, Body
from pydantic import BaseModel
import uuid
import asyncio
import os
import re
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.messaging_response import MessagingResponse
from groq import AsyncGroq

# Import your custom logic
from pc_rag_ingestion import sync_data
from pc_rag_retrieval import get_agent
from langchain_groq import ChatGroq

app = FastAPI()

# Enable CORS for the Chat-Bot
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL STATE ---
agent = None
page_index = None
is_ready = False

# Initialize Groq Client for WhatsApp
# (Chat-Bot uses ChatGroq via LangChain, WhatsApp uses AsyncGroq directly)
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))


class Query(BaseModel):
    question: str
    session_id: str = None


# --- INITIALIZATION LOGIC ---

async def initialize_rag_system():
    """
    Performs heavy lifting in background.
    Updates global variables so both Chat-Bot and WhatsApp can see them.
    """
    global agent, page_index, is_ready
    try:
        print("🚀 Background Task: Syncing data and loading RAG...")

        # We ensure sync_data result is assigned to the global page_index
        page_index = sync_data(reset=False)

        # Initialize the LangChain agent for the Chat-Bot
        agent = get_agent(
            llm=ChatGroq(model="llama-3.1-8b-instant"),
            page_index=page_index,
            domain="@alliedbenefit.com",
            key="allied"
        )

        is_ready = True
        print("✅ Background Task: RAG System fully loaded and ready.")
    except Exception as e:
        print(f"❌ Background Task Failed: {e}")


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(initialize_rag_system())


# --- FORMATTING HELPERS ---

def format_for_whatsapp(text: str) -> str:
    """Converts Markdown to WhatsApp-friendly syntax and cleans up leaks."""
    # 1. Clean up any 'leaked' citations the LLM might have hallucinated/copied
    # Removes patterns like [Source 1], (Email 2), Source 3:, etc.
    text = re.sub(r'\[?(Source|Email|SOURCE|EMAIL)\s*\d+\]?:?', '', text)

    # 2. **bold** -> *bold*
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)

    # 3. ### Header -> *Header* (WhatsApp doesn't have headers, so we bold them)
    text = re.sub(r'### (.*)', r'*\1*', text)

    # 4. Bullet points: Ensure they use a standard dash or dot
    text = text.replace("- ", "• ")

    return text.strip()


# --- ENDPOINTS ---

@app.get("/")
async def health_check():
    return {
        "status": "online",
        "rag_ready": is_ready
    }


@app.post("/ask")
async def ask(query: Query):
    """Chat-Bot Endpoint."""
    global agent, is_ready

    if not is_ready or agent is None:
        return {
            "answer": "The AI Assistant is currently syncing documents. Please try again in a moment.",
            "session_id": query.session_id,
            "status": "loading"
        }

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


# --- UPDATED FORMATTING HELPER ---

def format_for_whatsapp(text: str) -> str:
    """Converts Markdown to WhatsApp-friendly syntax and strips citations."""
    # Remove common citation patterns like [Source 1], (Source 1), or Source 1:
    text = re.sub(r'\[?Source\s*\d+\]?:?', '', text, flags=re.IGNORECASE)

    # **bold** -> *bold*
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    # ### Header -> *Header*
    text = re.sub(r'### (.*)', r'*\1*', text)
    # Convert bullet points if they aren't already supported
    text = text.replace("- ", "• ")

    return text.strip()


# --- UPDATED WHATSAPP LLM CALL ---

async def generate_whatsapp_llm_answer(query: str, context: str):
    """Direct Groq call for WhatsApp with strict 'No Reasoning' rule."""
    try:
        chat_completion = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a high-precision enterprise assistant. "
                        "STRICT RULES:\n"
                        "1. Provide ONLY the final answer. Never explain your search process.\n"
                        "2. Never say 'I found this in the column' or 'Based on the context'.\n"
                        "3. Use *bold* for key details (WhatsApp style).\n"
                        "4. If info is missing, say: 'Documentation does not specify the availability for this request.'\n"
                        "5. Do NOT mention source numbers or filenames."
                    )
                },
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ],
            temperature=0.1  # Lower temperature = less rambling
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return "I'm sorry, I'm having trouble processing that right now."


@app.post("/whatsapp")
async def whatsapp_reply(Body: str = Form(...), From: str = Form(...)):
    """Twilio WhatsApp Endpoint."""
    global page_index, is_ready

    resp = MessagingResponse()

    if not is_ready or page_index is None:
        resp.message("The assistant is still loading the latest rules. Please try again in 30 seconds!")
        return Response(content=str(resp), media_type="application/xml")

    try:
        user_query = Body.strip()

        # 1. Use the global page_index to search
        search_results = page_index.search(user_query, k=5)
        context_text = "\n---\n".join([doc.page_content for doc in search_results])

        # 2. Generate answer via Groq
        ai_response = await generate_whatsapp_llm_answer(user_query, context_text)

        # 3. Format and send
        final_text = format_for_whatsapp(ai_response)
        resp.message(final_text)

    except Exception as e:
        print(f"❌ WhatsApp Bridge Error: {e}")
        resp.message("I'm sorry, I encountered an error. Please try again.")

    return Response(content=str(resp), media_type="application/xml")


@app.post("/admin/reset-index")
async def reset_rag_index():
    global page_index, is_ready, agent
    try:
        is_ready = False
        persist_dir = os.getenv("PERSIST_DIRECTORY", "pc_page_index_db")
        index_path = os.path.join(persist_dir, "page_index.pkl")

        if os.path.exists(index_path):
            os.remove(index_path)

        # Re-run the initialization logic
        await initialize_rag_system()

        return {"status": "success", "message": "Index wiped and rebuilt successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}