FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ curl git libsqlite3-dev && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev,all-providers]"
COPY . .
# --timeout-keep-alive 120: heavy Python work (tree-sitter / networkx / leiden)
#   runs in asyncio.to_thread but still holds the GIL for hundreds of ms at a time.
#   With the default 5 s keep-alive, the proxy in the web container saw frequent
#   ECONNRESET ("socket hang up") whenever uvicorn closed an idle pooled
#   connection right before Node tried to reuse it during a GIL-busy window.
# --backlog 2048: keep more pending TCP connections queued so they aren't RST'd
#   by the kernel when the event loop is briefly starved.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", \
     "--timeout-keep-alive", "120", \
     "--backlog", "2048", \
     "--reload", "--reload-dir", "app", "--reload-include", "*.py"]
