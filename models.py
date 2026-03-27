import os
import re
import math
import pickle
from typing import List, Dict
from langchain_core.documents import Document

class PageIndex:
    def __init__(self):
        self.documents: List[Document] = []
        self.inverted_index: Dict[str, List[int]] = {}
        self.file_hashes: Dict[str, str] = {}
        self.avg_dl: float = 0.0
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been',
            'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'shall', 'can', 'this',
            'that', 'these', 'those', 'it', 'its', 'as', 'by', 'from',
            'not', 'no', 'so', 'if', 'we', 'they', 'he', 'she', 'you'
        }

    def __len__(self):
        return len(self.documents)

    @classmethod
    def exists(cls, path: str) -> bool:
        return os.path.exists(path)

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'\b[a-zA-Z0-9]{2,}\b', text.lower())
        return [t for t in tokens if t not in self.stopwords]

    def is_changed(self, identifier: str, current_hash: str) -> bool:
        return self.file_hashes.get(identifier) != current_hash

    def add_documents(self, docs: List[Document], identifier: str, new_hash: str):
        self.documents.extend(docs)
        self.file_hashes[identifier] = new_hash
        self._rebuild_inverted_index()

    def delete_by_source(self, full_path: str):
        self.documents = [d for d in self.documents if d.metadata.get("full_path") != full_path]
        self.file_hashes.pop(full_path, None)
        self._rebuild_inverted_index()

    def _rebuild_inverted_index(self):
        self.inverted_index = {}
        total_tokens = 0
        for i, doc in enumerate(self.documents):
            tokens = self._tokenize(doc.page_content)
            total_tokens += len(tokens)
            for token in set(tokens):
                self.inverted_index.setdefault(token, []).append(i)
        if self.documents:
            self.avg_dl = total_tokens / len(self.documents)

    def search(self, query: str, k: int = 30) -> List[Document]:
        q_tokens = self._tokenize(query)
        if not q_tokens or not self.documents:
            return self.documents[:k]
        scores: Dict[int, float] = {}
        N = len(self.documents)
        k1, b = 1.5, 0.75
        for token in q_tokens:
            posting_list = self.inverted_index.get(token, [])
            if not posting_list: continue
            n = len(posting_list)
            idf = math.log((N - n + 0.5) / (n + 0.5) + 1.0)
            for doc_idx in posting_list:
                doc_tokens = self._tokenize(self.documents[doc_idx].page_content)
                tf = doc_tokens.count(token)
                doc_len = len(doc_tokens)
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