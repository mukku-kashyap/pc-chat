import os
import json
import hashlib
from urllib.parse import urlparse
from settings import ENABLE_EMAIL, PERSIST_DIRECTORY, RESET_VECTOR_DB, DOCS_FOLDER

#HASH_STORE_PATH = "file_hashes.json"
# PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "claude_page_index_db")
# HASH_STORE_PATH = os.path.join(PERSIST_DIRECTORY, "file_hashes.json")

def is_remote_url(path):
    """Check if the path is a URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False

def compute_file_hash(file_path):
    """Compute SHA256 hash of a file OR a URL string."""
    sha256 = hashlib.sha256()

    # CASE 1: It's a URL
    if is_remote_url(file_path):
        # We hash the URL string itself.
        # Note: This only detects if the URL changed, not the content on the site.
        sha256.update(file_path.encode('utf-8'))
        return sha256.hexdigest()

    # CASE 2: It's a local file
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    return None

def load_hash_store():
    if os.path.exists(HASH_STORE_PATH):
        with open(HASH_STORE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_hash_store(hash_store):
    with open(HASH_STORE_PATH, "w") as f:
        json.dump(hash_store, f, indent=2)

def should_process_file(file_path):
    """
    Returns True if file changed or new.
    """
    hash_store = load_hash_store()
    current_hash = compute_file_hash(file_path)

    if file_path not in hash_store:
        hash_store[file_path] = current_hash
        save_hash_store(hash_store)
        return True

    if hash_store[file_path] != current_hash:
        hash_store[file_path] = current_hash
        save_hash_store(hash_store)
        return True

    return False
