"""
Script to fetch or verify technical documentation files for the RAG corpus.
Corpus source: Official FastAPI Documentation (Tutorial & Security guides).
"""
import os
import httpx
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

CORPUS_URLS = {
    "fastapi_dependency_injection.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/dependencies/index.md",
    "fastapi_background_tasks.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/background-tasks.md",
    "fastapi_security_oauth2.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/security/oauth2-jwt.md",
    "fastapi_middleware_cors.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/cors.md",
}

def verify_or_download_corpus():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Checking corpus directory: {DATA_DIR}")
    for filename, url in CORPUS_URLS.items():
        file_path = DATA_DIR / filename
        if file_path.exists():
            print(f" [EXISTS] {filename} ({file_path.stat().st_size} bytes)")
        else:
            print(f" [FETCHING] {filename} from {url}...")
            try:
                response = httpx.get(url, timeout=10.0, follow_redirects=True)
                if response.status_code == 200:
                    file_path.write_text(response.text, encoding="utf-8")
                    print(f" [SAVED] {filename} ({len(response.text)} characters)")
                else:
                    print(f" [ERROR] HTTP {response.status_code} fetching {url}")
            except Exception as e:
                print(f" [EXCEPTION] Failed to fetch {url}: {e}")

if __name__ == "__main__":
    verify_or_download_corpus()
