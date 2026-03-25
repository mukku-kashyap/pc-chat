from langchain_groq import ChatGroq
from pc_rag_ingestion import sync_data
from pc_rag_retrieval import get_agent
import pythoncom

llm = ChatGroq(model="llama-3.1-8b-instant")
page_index = sync_data(reset=False)

agent = get_agent(
    llm=llm,
    page_index=page_index,
    domain="@alliedbenefit.com",
    key="allied"
)

# session memory
sessions = {}

def rag_pipeline(question: str, session_id: str):
    pythoncom.CoInitialize()   # ✅ ADD THIS
    try:
        if session_id not in sessions:
            sessions[session_id] = {"configurable": {"thread_id": session_id}}
        config = sessions[session_id]
        result = agent.graph.invoke(
            {"input": question},
            config=config
        )
        return result["answer"]
    finally:
        pythoncom.CoUninitialize()   # ✅ CLEANUP