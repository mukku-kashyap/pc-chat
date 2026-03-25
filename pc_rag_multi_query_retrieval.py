import uuid
import os
import json
import builtins
import win32com.client
import hashlib
from datetime import datetime, timedelta
from typing import TypedDict, List, Annotated, Sequence, Optional
from dotenv import load_dotenv

# LangChain Core & Community
from langchain_core.messages import BaseMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereRerank
from langchain_groq import ChatGroq

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from pathlib import Path

# Assuming these are in the same directory
from pc_rag_ingestion import PageIndex, sync_data, PAGE_INDEX_PICKLE_PATH

load_dotenv()

# Formatting
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


# --- STATE DEFINITION ---

class AgentState(TypedDict):
    input: str  # The original/contextualized user input
    queries: List[str]  # The multi-query expansions
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: List[Document]
    answer: str


# --- RETRIEVER WRAPPER ---

class PageIndexRetriever(BaseRetriever):
    page_index: PageIndex
    k: int = 30

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.page_index.search(query, k=self.k)


# --- RAG AGENT ---

class RAGAgent:
    def __init__(self, llm, page_index: PageIndex, search_domain: str = None, search_key: str = None):
        self.llm = llm
        self.page_index = page_index
        self.search_domain = search_domain
        self.search_key = search_key

        # Retrieval Tools
        self.retriever = PageIndexRetriever(page_index=self.page_index, k=25)
        self.reranker = CohereRerank(cohere_api_key=os.environ["COHERE_API_KEY"], model="rerank-v3.5")
        self.outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")

        # Message Trimming (Context window management)
        self.trimmer = trim_messages(
            max_tokens=6000,
            strategy="last",
            token_counter=builtins.len,
            start_on="human",
            include_system=True
        )

        # Prompt Loading
        BASE_DIR = Path(__file__).resolve().parent
        PROMPT_PATH = BASE_DIR / "rag_system_prompt.txt"
        self.system_prompt = PROMPT_PATH.read_text(
            encoding="utf-8") if PROMPT_PATH.exists() else "You are a helpful assistant."

        self.graph = self._build_graph()

    # --- CORE NODES ---

    def contextualize_and_expand(self, state: AgentState):
        """Generates 3 search variations for better BM25 coverage."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a retrieval optimizer. Generate exactly 3 search queries in a JSON list based on the user's request. "
             "Vary the keywords to ensure high coverage in a keyword-search index. "
             "Output ONLY a JSON list of strings. Example: ['query 1', 'query 2', 'query 3']"),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
        ])

        chain = prompt | self.llm | StrOutputParser()
        try:
            res = chain.invoke({"input": state["input"], "messages": state["messages"]})
            # Clean possible markdown formatting
            clean_res = res.replace("```json", "").replace("```", "").strip()
            queries = json.loads(clean_res)
        except:
            queries = [state["input"]]  # Fallback to original

        return {"queries": queries}

    def retrieve_docs(self, state: AgentState):
        """Retrieves and deduplicates documents from PageIndex using multi-queries."""
        all_docs = []
        seen_paths = set()

        for q in state["queries"]:
            docs = self.retriever.invoke(q)
            for d in docs:
                path = d.metadata.get("full_path") or d.page_content[:50]
                if path not in seen_paths:
                    all_docs.append(d)
                    seen_paths.add(path)

        return {"context": all_docs}

    def retrieve_emails(self, state: AgentState):
        """Retrieves and deduplicates emails using keyword search."""
        raw_emails = self._fetch_outlook_emails()
        if not raw_emails:
            return {"context": state["context"]}

        # Create a temporary BM25 index for the fetched emails
        email_retriever = BM25Retriever.from_documents(raw_emails)
        email_retriever.k = 10

        combined_context = state["context"]
        seen_content_hashes = {hashlib.md5(d.page_content.encode()).hexdigest() for d in combined_context}

        for q in state["queries"]:
            email_docs = email_retriever.invoke(q)
            for ed in email_docs:
                h = hashlib.md5(ed.page_content.encode()).hexdigest()
                if h not in seen_content_hashes:
                    combined_context.append(ed)
                    seen_content_hashes.add(h)

        return {"context": combined_context}

    def rerank_docs(self, state: AgentState):
        """Uses Cohere to reorder the results by semantic relevance."""
        if not state["context"]:
            return {"context": []}

        reranked = self.reranker.compress_documents(
            documents=state["context"],
            query=state["input"]  # Rerank against the original question
        )
        # Limit to top 12 highest signal chunks
        return {"context": reranked[:12]}

    def generate_answer(self, state: AgentState):
        if not state["context"]:
            return {"answer": "I found no relevant documents to answer your question.",
                    "messages": [AIMessage(content="No info.")]}

        context_str = self._format_context(state["context"])
        today = datetime.now().strftime("%A, %B %d, %Y")

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{self.system_prompt}\n\nDate: {today}\n\nContext:\n{context_str}"),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
        ])

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"input": state["input"], "messages": state["messages"]})
        return {"answer": response, "messages": [AIMessage(content=response)]}

    # --- HELPERS ---

    def _fetch_outlook_emails(self) -> List[Document]:
        try:
            target_folders = [6, 5]  # Inbox, Sent
            raw_docs = []
            lookback = (datetime.now() - timedelta(days=60)).strftime("%m/%d/%Y %H:%M %p")

            for fid in target_folders:
                folder = self.outlook.GetDefaultFolder(fid)
                items = folder.Items.Restrict(f"[ReceivedTime] >= '{lookback}'")
                items.Sort("[ReceivedTime]", True)
                msg = items.GetFirst()

                while msg and len(raw_docs) < 50:
                    if getattr(msg, "Class", 0) == 43:  # MailItem
                        body = getattr(msg, 'Body', '')
                        sender = getattr(msg, 'SenderEmailAddress', '').lower()
                        subj = getattr(msg, 'Subject', '').lower()

                        # Match domain or keyword
                        if (self.search_key and self.search_key.lower() in (subj + body)) or \
                                (self.search_domain and self.search_domain.lower() in sender):
                            raw_docs.append(Document(
                                page_content=f"FROM: {sender}\nSUBJ: {subj}\nDATE: {msg.ReceivedTime}\nBODY: {body[:2000]}",
                                metadata={"is_email": True, "date": str(msg.ReceivedTime)}
                            ))
                    msg = items.GetNext()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            return splitter.split_documents(raw_docs)
        except:
            return []

    def _format_context(self, docs: List[Document]) -> str:
        formatted = []
        for i, d in enumerate(docs):
            src = d.metadata.get("full_path", "Email" if d.metadata.get("is_email") else "Doc")
            formatted.append(f"[{i + 1}] SOURCE: {src}\n{d.page_content}\n---")
        return "\n\n".join(formatted)

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("expand", self.contextualize_and_expand)
        workflow.add_node("retrieve_docs", self.retrieve_docs)
        workflow.add_node("retrieve_emails", self.retrieve_emails)
        workflow.add_node("rerank", self.rerank_docs)
        workflow.add_node("generate", self.generate_answer)

        workflow.add_edge(START, "expand")
        workflow.add_edge("expand", "retrieve_docs")
        workflow.add_edge("retrieve_docs", "retrieve_emails")
        workflow.add_edge("retrieve_emails", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile(checkpointer=MemorySaver())

def get_agent(llm, page_index: PageIndex, domain=None, key=None) -> RAGAgent:
    return RAGAgent(
        llm=llm,
        page_index=page_index,
        search_domain=domain,
        search_key=key,
    )

if __name__ == "__main__":
    llm = ChatGroq(model="llama-3.1-8b-instant")
    p_index = sync_data(reset=False)
    #agent = RAGAgent(llm, p_index, search_domain="@alliedbenefit.com", search_key="allied")
    my_agent = get_agent(
        llm=llm,
        page_index=p_index,
        domain="@alliedbenefit.com",
        key="allied"
    )
    sid = str(uuid.uuid4())
    while True:
        inp = input(f"\n{BLUE}You:{RESET} ")
        if inp.lower() in ["exit", "quit"]: break

        cfg = {"configurable": {"thread_id": sid}}
        res = my_agent.graph.invoke({"input": inp, "messages": []}, config=cfg)
        print(f"\n{GREEN}🤖:{RESET} {res['answer']}")