FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ curl git libsqlite3-dev && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev,all-providers]"
COPY . .
# --timeout-keep-alive 120: heavy Python work (tree-sitter / networkx / leiden)
#   now runs in a dedicated subprocess via wiki_runner, but we keep the long
#   keep-alive so proxies / the web container don't see ECONNRESET during any
#   brief GIL-busy windows on the uvicorn process itself.
# --backlog 2048: keep more pending TCP connections queued so they aren't RST'd
#   by the kernel when the event loop is briefly starved.
# Note: uvicorn ignores --workers when --reload is set.  We intentionally run a
# single reloadable process; heavy work is offloaded to subprocesses so one
# worker stays responsive regardless of repo size.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "120", \
     "--backlog", "2048", \
     "--reload", "--reload-dir", "app", "--reload-include", "*.py"]
