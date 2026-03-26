import re
import hashlib
import json
import shutil
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from unstructured.chunking.title import chunk_by_title
import sys
import os
from tqdm import tqdm
from extract_url import read_url_file_from_repository
from collections import defaultdict
import warnings
import pickle
import math
from unstructured.partition.auto import partition
import requests
from unstructured.partition.html import partition_html
from settings import ENABLE_EMAIL, PERSIST_DIRECTORY, RESET_VECTOR_DB, DOCS_FOLDER
import time

GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

# -------------------------------------------------------------------
# ⚙️ CONFIG
# -------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*max_size.*parameter is deprecated.*")
load_dotenv()
warnings.filterwarnings("ignore")

#DOCS_FOLDER = os.getenv("DOCS_FOLDER")
#PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY")
#RESET_VECTOR_DB = os.getenv("RESET_VECTOR_DB", "false").lower() == "true"

PAGE_INDEX_PICKLE_PATH = os.path.join(PERSIST_DIRECTORY, "page_index.pkl")

llm = ChatOpenAI(
    model="nvidia/nemotron-nano-12b-v2-vl:free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0.0,
)


# -------------------------------------------------------------------
# 🗂️ PAGE INDEX: In-memory index with disk persistence
# -------------------------------------------------------------------
class PageIndex:

    def __init__(self):
        self.documents: List[Document] = []
        # Maps token -> list of document indices
        self.inverted_index: Dict[str, List[int]] = {}
        # Maps source (path/URL) -> SHA-256 hash string
        self.file_hashes: Dict[str, str] = {}
        # Pre-calculated for O(1) retrieval performance
        self.avg_dl: float = 0.0

        # Optimized stopword set for higher search signal
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been',
            'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'shall', 'can', 'this',
            'that', 'these', 'those', 'it', 'its', 'as', 'by', 'from',
            'not', 'no', 'so', 'if', 'we', 'they', 'he', 'she', 'you'
        }

    def __len__(self):
        """Allows len(page_index) to return the number of documents."""
        return len(self.documents)

    @classmethod
    def exists(cls, path: str) -> bool:
        return os.path.exists(path)

    def _tokenize(self, text: str) -> List[str]:
        """Fast regex tokenizer with stopword removal."""
        tokens = re.findall(r'\b[a-zA-Z0-9]{2,}\b', text.lower())
        return [t for t in tokens if t not in self.stopwords]

    def is_changed(self, identifier: str, current_hash: str) -> bool:
        """Checks if a file or URL hash differs from the stored version."""
        return self.file_hashes.get(identifier) != current_hash

    def add_documents(self, docs: List[Document], identifier: str, new_hash: str):
        """Adds new documents and immediately updates the inverted index."""
        self.documents.extend(docs)
        self.file_hashes[identifier] = new_hash
        self._rebuild_inverted_index()

    def delete_by_source(self, full_path: str):
        """Removes all documents from a source without manual list rebuilding."""
        self.documents = [d for d in self.documents if d.metadata.get("full_path") != full_path]
        self.file_hashes.pop(full_path, None)
        self._rebuild_inverted_index()

    def _rebuild_inverted_index(self):
        """Optimized rebuilding of the search index and global stats."""
        self.inverted_index = {}
        total_tokens = 0

        for i, doc in enumerate(self.documents):
            tokens = self._tokenize(doc.page_content)
            total_tokens += len(tokens)
            # Use set for the inverted index to avoid duplicate pointers per doc
            for token in set(tokens):
                self.inverted_index.setdefault(token, []).append(i)

        if self.documents:
            self.avg_dl = total_tokens / len(self.documents)

    def search(self, query: str, k: int = 30) -> List[Document]:
        """Professional BM25 retrieval logic."""
        q_tokens = self._tokenize(query)
        if not q_tokens or not self.documents:
            return self.documents[:k]

        scores: Dict[int, float] = {}
        N = len(self.documents)
        k1, b = 1.5, 0.75

        for token in q_tokens:
            posting_list = self.inverted_index.get(token, [])
            if not posting_list: continue

            # IDF: Rare words get higher weight
            n = len(posting_list)
            idf = math.log((N - n + 0.5) / (n + 0.5) + 1.0)

            for doc_idx in posting_list:
                doc_tokens = self._tokenize(self.documents[doc_idx].page_content)
                tf = doc_tokens.count(token)
                doc_len = len(doc_tokens)

                # BM25 formula for document-level normalization
                denom = tf + k1 * (1 - b + b * (doc_len / self.avg_dl))
                scores[doc_idx] = scores.get(doc_idx, 0.0) + (idf * (tf * (k1 + 1) / denom))

        sorted_results = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return [self.documents[i] for i in sorted_results[:k]]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "PageIndex":
        if cls.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return cls()

    def get_all_documents(self) -> List[Document]:
        return list(self.documents)


def get_jina_content(url: str) -> str:
    """Uses Jina Reader to fetch clean, rendered content from a URL."""
    try:
        # The 'r.jina.ai' prefix tells Jina to fetch and process the URL
        jina_url = f"https://r.jina.ai/{url}"

        # We use a standard timeout to prevent the script from hanging
        response = requests.get(jina_url, timeout=20)
        time.sleep(2)

        if response.status_code == 200:
            content = response.text
            return content
        else:
            print(f"⚠️ Jina returned status code: {response.status_code}")
            return ""
    except Exception as e:
        print(f"❌ Jina request failed for {url}: {e}")
        return ""

def get_file_hash(filepath: str) -> str:
    """Reads a local file and returns its SHA-256 hash."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        print(f"❌ Could not hash file {filepath}: {e}")
        return ""

def partition_document(full_path: str, html_str: str = None):
    elements = []  # Initialize as empty list
    if full_path.startswith("http"):
        if not html_str or len(html_str.strip()) < 500:
            print(f"⚠️ Skipping {full_path}: HTML content is missing or too short.")
            return []
        try:
            elements = partition_html(
                text=html_str,
                strategy="hi_res",
                infer_table_structure=True,
                languages=["eng"]
            )
        except Exception as e:
            print(f"❌ Unstructured failed to parse HTML for {full_path}: {e}")
            return []

    else:
        try:
            elements = partition(
                filename=full_path,
                strategy="hi_res",
                infer_table_structure=True,
                extract_image_block_to_payload=True,
                languages=["eng"],
            )
        except Exception as e:
            print(f"❌ Failed to partition file {full_path}: {e}")
            return []

    return elements

# -------------------------------------------------------------------
# ✂️ STEP 2: TITLE-BASED CHUNKING
# -------------------------------------------------------------------
def chunk_elements(elements):
    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )
    return chunks


# -------------------------------------------------------------------
# 🔎 STEP 3: SEPARATE CONTENT TYPES
# -------------------------------------------------------------------
def separate_content_types(chunk) -> Dict:
    content_data = {
        "text": chunk.text or "",
        "tables": [],
        "images": [],
        "types": ["text"],
    }

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            elif element_type == "Image":
                if hasattr(element.metadata, "image_base64"):
                    img_str = element.metadata.image_base64
                    if img_str and len(img_str) > 7000:
                        content_data["types"].append("image")
                        content_data["images"].append(img_str)

    content_data["types"] = list(set(content_data["types"]))
    return content_data


# -------------------------------------------------------------------
# 🧠 STEP 4: AI-ENHANCED SUMMARY (ONLY WHEN NEEDED)
# -------------------------------------------------------------------
def create_ai_summary(text: str, tables: List[str], images: List[str]) -> str:
    system_message = SystemMessage(
        content=(
            "You are an AI assistant that creates highly searchable, retrieval-optimized "
            "descriptions for document chunks.\n\n"
            "Your output will be stored in a document index.\n"
            "PRIORITIES:\n"
            "- Maximize semantic recall\n"
            "- Include facts, entities, numbers, and relationships\n"
            "- Mention what questions this content can answer\n"
            "- Describe visual patterns if images are present\n"
            "- Include alternative phrasings and keywords\n"
            "- Be descriptive, NOT conversational\n"
            "- Do NOT reference the word 'document' or 'chunk'"
        )
    )

    prompt_text = "CONTENT TO ANALYZE:\n\n"
    if text.strip():
        prompt_text += f"TEXT:\n{text}\n\n"

    if tables:
        prompt_text += "TABLES (HTML):\n"
        for t in tables:
            prompt_text += f"{t}\n\n"

    prompt_text += (
        "TASK:\n"
        "Generate a detailed, searchable description that includes:\n"
        "1. Key facts, numbers, entities, and metrics\n"
        "2. Main topics and concepts\n"
        "3. Questions this content could answer\n"
        "4. Insights derived from tables\n"
        "5. Visual interpretation of images (charts, diagrams, layouts)\n"
        "6. Alternative terms and synonyms users may search for\n\n"
        "SEARCHABLE DESCRIPTION:"
    )
    human_content = [{"type": "text", "text": prompt_text}]

    for img in images:
        if img and len(img) > 100:
            human_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
            })

    human_message = HumanMessage(content=human_content)
    try:
        response = llm.invoke([system_message, human_message])
    except Exception as e:
        return text

    return response.content.strip()


# -------------------------------------------------------------------
# 📦 STEP 5: CONVERT CHUNKS → LANGCHAIN DOCUMENTS
# -------------------------------------------------------------------
def summarise_chunks_as_documents_for_pageindex(chunks, full_path, display_name):
    documents = []

    for chunk in chunks:
        content_data = separate_content_types(chunk)
        if (
            len(content_data["text"].strip()) < 50
            and not content_data["tables"]
            and not content_data["images"]
        ):
            continue

        # Only use AI if tables or images exist
        if content_data["tables"] or content_data["images"]:
            enhanced_text = create_ai_summary(
                content_data["text"],
                content_data["tables"],
                content_data["images"]
            )
        else:
            enhanced_text = content_data["text"]

        sheet_names = set()
        table_count = 0
        if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
            for el in chunk.metadata.orig_elements:
                sheet = getattr(el.metadata, "sheet_name", None)
                if sheet:
                    sheet_names.add(sheet)
                if type(el).__name__ == "Table":
                    table_count += 1

        metadata = {
            "source_file": full_path,
            "folder": os.path.dirname(display_name),
            "full_path": full_path,
            "sheet_names": list(sheet_names) if sheet_names else None,
            "contains_table": table_count > 0,
            "table_count": table_count,
            "contains_image": bool(content_data["images"]),
            "content_types": content_data["types"],
            "original_content": json.dumps(content_data),
        }

        doc = Document(
            page_content=enhanced_text[:3000],
            metadata=metadata,
        )
        documents.append(doc)

    return documents


# -------------------------------------------------------------------
# 🚀 PROCESS A SINGLE SOURCE
# -------------------------------------------------------------------
def process_source(full_path, display_name, html_str=None):
    with tqdm(total=100,
              file=sys.stdout,
              desc=f"{BLUE}{BOLD}📂 Processing {display_name}{RESET}",
              bar_format="{desc}: {percentage:3.0f}%|{bar}|",
              leave=True) as pbar:

        raw_elements = partition_document(full_path, html_str=html_str)
        pbar.update(30)

        grouped_elements = isolate_structural_groups(raw_elements)
        chunks = []
        for group in grouped_elements:
            grouped_chunks = chunk_elements(group)
            chunks.extend(grouped_chunks)
        pbar.update(30)

        docs = summarise_chunks_as_documents_for_pageindex(chunks, full_path, display_name)
        pbar.update(40)

    print(f"{GREEN}  ✓ Extracted {len(raw_elements)} elements{RESET}")
    print(f"{GREEN}  ✓ Created {len(chunks)} chunks{RESET}")
    print(f"{GREEN}  ✓ Final docs: {len(docs)}{RESET}")

    return docs

def isolate_structural_groups(elements):
    """
    Detect if elements contain sheet structure (Excel).
    If yes → group by sheet. Else → return single group.
    """
    has_sheet_info = any(
        getattr(el.metadata, "sheet_name", None)
        for el in elements
    )

    if not has_sheet_info:
        return [elements]

    grouped = defaultdict(list)
    for el in elements:
        sheet = getattr(el.metadata, "sheet_name", None)
        grouped[sheet].append(el)

    return list(grouped.values())

def sync_data(reset=False) -> PageIndex:
    # 1. Setup & Reset Logic
    if reset and os.path.exists(PERSIST_DIRECTORY):
        if os.path.exists(PAGE_INDEX_PICKLE_PATH):
            os.remove(PAGE_INDEX_PICKLE_PATH)
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)

        # file_hashes.json is no longer needed as it's inside the pickle now
        print(f"{BLUE}🧹 Resetting index...{RESET}")

    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    # Load existing index OR start a fresh one
    if PageIndex.exists(PAGE_INDEX_PICKLE_PATH) and not reset:
        print(f"📂 Loading existing index from {PAGE_INDEX_PICKLE_PATH}...")
        page_index = PageIndex.load(PAGE_INDEX_PICKLE_PATH)
    else:
        state = "Resetting" if reset else "Initializing new"
        print(f"✨ {state} index...")
        page_index = PageIndex()

    # 2. Collect ALL Sources (Files + URLs)
    all_sources = []  # List of (full_path_or_url, display_name, is_url)

    # Add Local Files
    if os.path.exists(DOCS_FOLDER):
        for root, _, files in os.walk(DOCS_FOLDER):
            for file in files:
                if file.lower() == "urls.txt": continue
                full_path = os.path.join(root, file)
                display_name = os.path.relpath(full_path, DOCS_FOLDER).replace("\\", "/")
                all_sources.append((full_path, display_name, False))

    # Add URLs
    urls = read_url_file_from_repository(DOCS_FOLDER)
    if urls:
        for url in urls:
            # For URLs, path and display name are the same
            all_sources.append((url, url, True))

    # 3. Process with Internal Change Detection
    has_updates = False

    # Identify and remove "Ghost" files (files in index that no longer exist on disk/list)
    current_source_names = {s[1] for s in all_sources}
    for indexed_name in list(page_index.file_hashes.keys()):
        if indexed_name not in current_source_names:
            print(f"🗑️ Source removed from disk, deleting from index: {indexed_name}")
            page_index.delete_by_source(indexed_name)
            has_updates = True

    for full_path, display_name, is_url in all_sources:
        if is_url:
            # --- URL OPTIMIZED PATH ---
            current_html = get_jina_content(full_path)
            time.sleep(2)
            if not current_html or len(current_html.strip()) < 500:
                print(f"⚠️ Skipping {display_name}: Content too short or timeout.")
                continue

            new_hash = hashlib.sha256(current_html.encode('utf-8')).hexdigest() if current_html else ""
            if page_index.file_hashes.get(full_path) != new_hash:
                print(f"🔄 New URL Processing: {display_name}")
                page_index.delete_by_source(display_name)

                # Pass the HTML we ALREADY have
                docs = process_source(full_path, display_name, html_str=current_html)
                if docs:
                    # Update docs and store the hash we already calculated
                    page_index.add_documents(docs, display_name, new_hash)
                    has_updates = True
            else:
                print(f"⏭ Skipping unchanged URL: {display_name}")
        else:
            file_hash = get_file_hash(full_path)
            if page_index.is_changed(display_name, file_hash):
                print(f"🔄 File Changed Processing: {display_name}")
                page_index.delete_by_source(display_name)
                try:
                    extracted_docs = process_source(full_path, display_name, html_str=None)
                    if extracted_docs:
                        # 3. Add to index (this handles the new hash and rebuilds inverted index)
                        page_index.add_documents(extracted_docs, display_name, file_hash)
                        has_updates = True
                except Exception as e:
                    print(f"❌ Error processing {display_name}: {e}")
            else:
                print(f"⏭ Skipping unchanged file: {display_name}")
                pass


    if has_updates:
        page_index.save(PAGE_INDEX_PICKLE_PATH)
        print(f"{GREEN}✅ Sync Complete. Total docs in index: {len(page_index)}{RESET}")
    else:
        print(f"{GREEN}✅ Everything up to date.{RESET}")

    return page_index

if __name__ == "__main__":
    page_index = sync_data(reset=RESET_VECTOR_DB)




