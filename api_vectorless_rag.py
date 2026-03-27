from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import asyncio
from pc_rag_ingestion import sync_data
from pc_rag_retrieval import get_agent
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware
import os
from models import PageIndex
from twilio.twiml.messaging_response import MessagingResponse
import re

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for the background state
agent = None
is_ready = False


class Query(BaseModel):
    question: str
    session_id: str = None


async def initialize_rag_system():
    """
    Performs the heavy lifting in the background so the
    web server port can open immediately.
    """
    global agent, is_ready
    try:
        print("🚀 Background Task: Syncing data and loading RAG...")
        # This is the part that takes a long time
        page_index = sync_data(reset=False)

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
    """
    FastAPI startup hook. We trigger the background task
    but do NOT 'await' it, allowing the port to bind immediately.
    """
    asyncio.create_task(initialize_rag_system())


@app.get("/")
async def health_check():
    """
    Endpoint for Render to verify the service is live.
    """
    return {
        "status": "online",
        "rag_ready": is_ready
    }


@app.post("/ask")
async def ask(query: Query):
    global agent, is_ready

    # Safety check in case a user queries before indexing finishes
    if not is_ready or agent is None:
        return {
            "answer": "The AI Assistant is currently syncing documents and will be ready in a moment. Please try again shortly.",
            "session_id": query.session_id,
            "status": "loading"
        }

    session_id = query.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    # Invoke the graph
    result = agent.graph.invoke(
        {"input": query.question},
        config=config
    )

    return {
        "answer": result["answer"],
        "session_id": session_id
    }


@app.post("/admin/reset-index")
async def reset_rag_index():
    # 1. Get paths from environment
    persist_dir = os.getenv("PERSIST_DIRECTORY", "pc_page_index_db")
    index_path = os.path.join(persist_dir, "page_index.pkl")

    global page_index, is_ready
    try:
        is_ready = False  # Block chat requests during rebuild

        # 2. Physical Cleanup
        if os.path.exists(index_path):
            os.remove(index_path)
            print(f"🗑️ Deleted old index at {index_path}")

        # 3. Re-initialize the object from our new models.py
        from models import PageIndex
        page_index = PageIndex()

        # 4. Trigger the sync logic
        # IMPORTANT: Ensure your sync_data function in pc_rag_ingestion
        # is designed to return the docs or update a passed index.
        from pc_rag_ingestion import sync_data
        await sync_data()

        # 5. Reload the freshly saved index into the API's memory
        page_index = PageIndex.load(index_path)

        is_ready = True
        return {"status": "success", "message": "Index wiped and rebuilt successfully."}
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return {"status": "error", "message": str(e)}


def format_for_whatsapp(text: str) -> str:
    """
    Converts standard Markdown to WhatsApp formatting.
    Example: **text** becomes *text*
    """
    # Replace double asterisks with single asterisks for bold
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    # Replace __text__ or _text_ with _text_ for italics (already standard)
    return text


@app.post("/whatsapp")
async def whatsapp_reply(Body: str = Form(...), From: str = Form(...)):
    """
    Handles incoming WhatsApp messages from Twilio.
    - Body: The text the user sent.
    - From: The user's WhatsApp number (e.g., 'whatsapp:+919876543210').
    """
    resp = MessagingResponse()

    # 1. Check if RAG is actually ready
    if not is_ready or page_index is None:
        resp.message("System is currently updating its knowledge. Please try again in a minute.")
        return Response(content=str(resp), media_type="application/xml")

    try:
        user_query = Body.strip()

        # 2. Search your indexed PDF/Web data
        # We use k=5 to get a good amount of context for the AI
        search_results = page_index.search(user_query, k=5)

        # Combine the content of the search results into one string
        context_text = "\n---\n".join([doc.page_content for doc in search_results])

        # 3. Generate Answer using your LLM logic
        # Replace 'llm_chain' with whatever your LLM variable name is
        # If you're using direct OpenAI, call your completion function here
        ai_response = await generate_llm_answer(user_query, context_text)

        # 4. Format for WhatsApp (Markdown to Asterisks)
        final_text = format_for_whatsapp(ai_response)

        # 5. Build Twilio Response
        resp.message(final_text)

    except Exception as e:
        print(f"❌ WhatsApp Error: {e}")
        resp.message("I encountered a small glitch while searching. Could you try rephrasing that?")

    return Response(content=str(resp), media_type="application/xml")


async def generate_llm_answer(query, context):
    """
    Helper to call your LLM.
    Using a standard prompt structure for your PC project.
    """
    prompt = f"""
    You are the Princess Cottage Assistant. Use the context below to answer the question.
    If the answer isn't in the context, say you don't know politely.

    Context: {context}

    Question: {query}
    Answer:
    """
    # Replace this with your specific OpenAI/LLM call logic
    # Example: response = client.chat.completions.create(...)
    # return response.choices[0].message.content
    pass  # You will put your specific LLM variable call here