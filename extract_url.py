import re
from pathlib import Path
from typing import List
from urllib.parse import urlparse

def read_urls_from_file(file_path: str) -> List[str]:
    path = Path(file_path)/ "urls.txt"

    if not path.exists():
        return None

    content = path.read_text(encoding="utf-8")
    raw_strings = re.split(r"[;, \n]", content)

    urls = [url.strip() for url in raw_strings if url.strip()]
    valid_urls = [url for url in urls if is_valid_url(url)]

    if not valid_urls:
        return None

    #print(f"🌐 Found {len(list(dict.fromkeys(valid_urls)))} unique and valid URLs.")
    return list(dict.fromkeys(valid_urls))

def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return all([parsed.scheme in ("http", "https"), parsed.netloc])


def read_url_file_from_repository(file_path: str) -> List[str]:
    urls = read_urls_from_file(file_path)
    if urls:
        return [url for url in urls if is_valid_url(url)]
    return None


#print(read_url_file_from_repository("url"))