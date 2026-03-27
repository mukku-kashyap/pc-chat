import uuid
import os
import builtins
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
from models import PageIndex

from pc_rag_ingestion import sync_data
from pathlib import Path
from settings import ENABLE_EMAIL

load_dotenv()
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

#ENABLE_EMAIL = os.getenv("ENABLE_EMAIL", "false").lower() == "true"
# --- STEP 1: STATE DEFINITION ---

class AgentState(TypedDict):
    input: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: List[Document]
    answer: str


# --- STEP 2: PageIndex-backed Retriever (LangChain Compatible) ---

class PageIndexRetriever(BaseRetriever):
    """
    LangChain-compatible retriever backed by PageIndex.
    Uses BM25 keyword search over the in-memory page index.
    """

    page_index: PageIndex
    k: int = 30

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.page_index.search(query, k=self.k)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


# --- STEP 3: RAG AGENT CLASS ---

class RAGAgent:
    def __init__(
            self,
            llm,
            page_index: PageIndex,
            search_domain: Optional[str] = None,
            search_key: Optional[str] = None,
    ):
        if page_index is None or len(page_index) == 0:
            raise ValueError("❌ Initialization Error: PageIndex is empty or not provided.")

        self.llm = llm
        self.page_index = page_index
        self.search_domain = search_domain
        self.search_key = search_key

        self.retriever = self._setup_retriever()
        self.email_bm25 = None
        self.email_chunks_cache = []
        self.trimmer = trim_messages(
            max_tokens=6000,
            strategy="last",
            token_counter=builtins.len,
            start_on="human",
            include_system=True
        )
        self.graph = self._build_graph()

        BASE_DIR = Path(__file__).resolve().parent
        PROMPT_PATH = BASE_DIR / "rag_system_prompt.txt"
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

        if ENABLE_EMAIL:
            import win32com.client
            self.outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")

        self.reranker = CohereRerank(
            cohere_api_key=os.environ["COHERE_API_KEY"],
            model="rerank-v3.5"
        )

    def _setup_retriever(self) -> BaseRetriever:
        # return PageIndexRetriever(page_index=self.page_index, k=40)
        """
        Builds a hybrid retriever:
        - Primary: PageIndex keyword search (inverted index)
        - Secondary: BM25Retriever over all documents for additional coverage
        Combined using EnsembleRetriever.
        """
        all_docs = self.page_index.get_all_documents()

        # Primary: PageIndex retriever (fast inverted index)
        pi_retriever = PageIndexRetriever(page_index=self.page_index, k=20)

        # Secondary: BM25 over all docs for complementary recall
        if all_docs and len(all_docs) > 0:
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = 20

            # Combine both retrievers
            try:
                from langchain_classic.retrievers import EnsembleRetriever
                base_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, pi_retriever],
                    weights=[0.4, 0.6]
                )
                print(f"{BLUE}🧬 Mode: HYBRID PageIndex + BM25 Enabled.{RESET}")
            except ImportError:
                # Fallback if langchain_classic not available
                base_retriever = bm25_retriever
                print(f"{BLUE}⌨️ Mode: BM25-ONLY (langchain_classic not found).{RESET}")
        else:
            base_retriever = pi_retriever
            print(f"{BLUE}📑 Mode: PageIndex-ONLY Enabled.{RESET}")

        return base_retriever

    def _rerank_combined(self, state: AgentState):
        docs = state.get("context", [])
        if not docs:
            return {"context": []}

        reranked = self.reranker.compress_documents(
            documents=docs,
            query=state["input"]
        )
        return {"context": reranked}

    def _get_email_documents(self, query: str) -> List[Document]:
        try:
            target_folders = [6, 5]  # Inbox and Sent
            raw_docs = []

            # 1. Broad Lookback: 60 days to ensure we don't miss anything
            start_date = (datetime.now() - timedelta(days=60)).strftime("%m/%d/%Y %H:%M %p")

            for folder_id in target_folders:
                folder = self.outlook.GetDefaultFolder(folder_id)
                # Use Restrict to filter by date first (fastest)
                items = folder.Items.Restrict(f"[ReceivedTime] >= '{start_date}'")
                items.Sort("[ReceivedTime]", True)

                msg = items.GetFirst()

                while msg and len(raw_docs) < 50:
                    try:
                        # Skip non-mail items (Meeting requests, etc.)
                        if getattr(msg, "Class", 0) != 43:
                            msg = items.GetNext()
                            continue

                        # Capture all text fields
                        subj = (getattr(msg, 'Subject', '') or "").lower()
                        body = (getattr(msg, 'Body', '') or "").lower()
                        sender = (getattr(msg, 'SenderEmailAddress', '') or "").lower()
                        to_rec = (getattr(msg, 'To', '') or "").lower()
                        cc_rec = (getattr(msg, 'CC', '') or "").lower()

                        # Target Keyword & Domain (Case-Insensitive)
                        key = self.search_key.lower() if self.search_key else "___none___"
                        dom = self.search_domain.lower() if self.search_domain else "___none___"

                        # --- THE "ANYWHERE" CHECK ---
                        # Check Subject, Body, Sender, To, and CC
                        if (key in subj or key in body or key in sender or key in to_rec or key in cc_rec) or \
                                (dom in sender or dom in to_rec or dom in cc_rec):
                            raw_docs.append(Document(
                                page_content=(
                                    f"SOURCE: {'INBOX' if folder_id == 6 else 'SENT'}\n"
                                    f"DATE: {msg.ReceivedTime}\n"
                                    f"FROM: {msg.SenderName} <{sender}>\n"
                                    f"TO: {to_rec}\n"
                                    f"SUBJ: {getattr(msg, 'Subject', '')}\n"
                                    f"BODY: {getattr(msg, 'Body', '')[:2500]}"  # Grab more body text
                                ),
                                metadata={"date": str(msg.ReceivedTime), "is_email": True}
                            ))
                    except Exception:
                        pass  # Silently skip corrupted items

                    msg = items.GetNext()

            #print(f"✅ Found {len(raw_docs)} Allied-related emails.")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            return splitter.split_documents(raw_docs)

        except Exception as e:
            print(f"❌ Outlook Error: {e}")
            return []

    # --- GRAPH NODES ---

    def contextualize_question(self, state: AgentState):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
                "You are a query rewriting assistant.\n"
                "Your task:\n"
                "Rewrite the user's input into a clear, standalone query while preserving its original intent.\n"
                "Rules:\n"
                "1. If the user input is already clear and standalone, return it unchanged.\n"
                "2. If it is a question, rewrite it as a grammatically correct standalone question.\n"
                "3. If it is a command or instruction (e.g., 'generate a summary report'), rewrite it as a clear standalone request.\n"
                "4. Use chat history only to resolve ambiguous references (it, they, this, that, 'the agreement').\n"
                "5. Preserve all specific entities (e.g., 'Inovaare', 'Allied').\n"
                "6. Do NOT refuse.\n"
                "7. Do NOT explain your reasoning.\n"
                "8. Output only the rewritten query."),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
        ])
        trimmed_msgs = self.trimmer.invoke(state["messages"])
        chain = prompt | self.llm | StrOutputParser()
        standalone = chain.invoke({"input": state["input"], "messages": trimmed_msgs})

        return {
            "input": standalone,
            "messages": state["messages"]
        }

    def retrieve_docs(self, state: AgentState):
        """Node 1: PageIndex retrieval."""
        docs = self.retriever.invoke(state["input"])
        return {"context": docs}

    def retrieve_emails(self, state: AgentState):
        """Node 2: Email retrieval."""
        if not ENABLE_EMAIL:
            #print("Skippiing Emails... ")
            return {"context": state.get("context", [])}
        #print(f"Retrieving emails for: {state.get('input')}")
        chunks = self._get_email_documents(state.get("input"))
        #print("Retrieved Email Chunks= ", chunks)
        if chunks:
            self.email_bm25 = BM25Retriever.from_documents(chunks)
            self.email_bm25.k = 20
            best_email_chunks = self.email_bm25.invoke(state["input"])
            #print("Email Chunks\n", best_email_chunks)

            #best_email_chunks = chunks[:20]

            current_context = state.get("context", [])
            return {"context": current_context + best_email_chunks}

        return {"context": state.get("context", [])}

    def _format_context(self, documents: List[Document]) -> str:
        emails = []
        docs_only = []

        for doc in documents:
            content = doc.page_content.strip()
            #if content.startswith("SOURCE:"):
            if doc.metadata.get("is_email"):
                emails.append(doc)
            else:
                docs_only.append(doc)

        # Sort emails newest first
        emails.sort(
            key=lambda d: d.metadata.get("date", ""),
            reverse=True
        )

        formatted_blocks = []

        for idx, doc in enumerate(emails, start=1):
            content = doc.page_content.strip()
            block = (
                f"EMAIL {idx}\n"
                f"{content}\n"
                f"--- END EMAIL {idx} ---"
            )
            formatted_blocks.append(block)

        for idx, doc in enumerate(docs_only, start=1):
            metadata = doc.metadata or {}
            content = doc.page_content.strip()
            source_file = metadata.get("source_file", "Unknown Source")
            sheet_names = metadata.get("sheet_names", "")
            contains_table = metadata.get("contains_table", False)

            block = (
                f"SOURCE {idx}\n"
                f"FILE: {source_file}\n"
                f"SHEETS: {sheet_names}\n"
                f"CONTAINS_TABLE: {contains_table}\n\n"
                f"{content}\n"
                f"--- END SOURCE {idx} ---"
            )
            formatted_blocks.append(block)



        return "\n\n".join(formatted_blocks)

    def generate_answer(self, state: AgentState):
        if not state.get("context"):
            return {
                "answer": "I do not have enough information in the retrieved sources to answer this.",
                "messages": [
                    AIMessage(content="I do not have enough information in the retrieved sources to answer this.")
                ]
            }

        today_date = datetime.now().strftime("%A, %B %d, %Y")
        MAX_CONTEXT_DOCS = 6 if len(state["context"]) > 10 else 8
        documents = state["context"][:MAX_CONTEXT_DOCS]
        context = self._format_context(documents)

        trimmed_msgs = self.trimmer.invoke(state["messages"])
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             self.system_prompt + "\n\n"
             "Today's date: {today_date}\n\n"
             "Use the context below to answer:\n\n"
             "{context}"),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
        ])

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "input": state["input"],
            "context": context,
            "messages": trimmed_msgs,
            "today_date": today_date
        })
        return {"answer": response, "messages": [AIMessage(content=response)]}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("contextualize", self.contextualize_question)
        workflow.add_node("retrieve_docs", self.retrieve_docs)
        workflow.add_node("retrieve_emails", self.retrieve_emails)
        workflow.add_node("rerank_combined", self._rerank_combined)
        workflow.add_node("generate", self.generate_answer)

        workflow.add_edge(START, "contextualize")
        workflow.add_edge("contextualize", "retrieve_docs")
        workflow.add_edge("retrieve_docs", "retrieve_emails")
        workflow.add_edge("retrieve_emails", "rerank_combined")
        workflow.add_edge("rerank_combined", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile(checkpointer=MemorySaver())


# --- HELPER ---

def get_agent(llm, page_index: PageIndex, domain=None, key=None) -> RAGAgent:
    return RAGAgent(
        llm=llm,
        page_index=page_index,
        search_domain=domain,
        search_key=key,
    )


# --- MAIN ---

if __name__ == "__main__":
    #PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "claude_page_index_db")
    llm = ChatGroq(model="llama-3.1-8b-instant")
    page_index = sync_data(reset=False)  # Just sync what's needed and move on
    my_agent = get_agent(
        llm=llm,
        page_index=page_index,
        domain="@alliedbenefit.com",
        key="allied"
    )

    session_id = str(uuid.uuid4())
    while True:
        user_input = input(f"\n{BLUE}{BOLD}You:{RESET} ")
        if user_input.lower() in ["exit", "quit"]:
            break

        config = {"configurable": {"thread_id": session_id}}
        result = my_agent.graph.invoke({"input": user_input}, config=config)
        print(f"\n{GREEN}{BOLD}🤖 Assistant:{RESET} {result['answer']}")



