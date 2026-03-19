# Wikis

AI-powered documentation generator that turns any code repository into a browsable, searchable wiki with architecture diagrams, code explanations, and an AI Q&A assistant.

## Features

- **Wiki generation** from GitHub, GitLab, Bitbucket, and Azure DevOps repositories
- **14+ language support** via tree-sitter parsing (Python, TypeScript, Java, Go, Rust, etc.)
- **AI Q&A** — ask questions about any repository with source-cited answers
- **Deep Research** — multi-step research engine for complex codebase questions
- **Mermaid diagrams** — auto-generated architecture and flow diagrams
- **MCP Server** — integrate with AI IDEs (Claude Code, Cursor, Windsurf)
- **Multiple LLM providers** — OpenAI, Anthropic, Ollama (local), Gemini, Bedrock
- **Self-hosted** — runs entirely on your infrastructure with Docker Compose
- **Authentication** — username/password, GitHub OAuth, Google OAuth via Better-Auth

## Quick Start

One command to install and run:

```bash
curl -fsSL https://raw.githubusercontent.com/arozumenko/wikis/main/install.sh | bash
```

This will clone the repo, prompt for your LLM provider/key, generate JWT keys, and start everything via Docker Compose.

Or manually (macOS / Linux / Windows):

```bash
git clone https://github.com/arozumenko/wikis.git && cd wikis
cp .env.example .env    # Edit: set LLM_PROVIDER and LLM_API_KEY
docker compose up -d
```

> **Windows:** The install script requires bash (WSL or Git Bash). Alternatively, clone the repo, copy `.env.example` to `.env`, fill in your LLM key, and run `docker compose up -d`.

Open [http://localhost:3000](http://localhost:3000) and log in with the default admin account:
**admin@wikis.dev** / **changeme123**

> Change the default password immediately after first login.

## Architecture

```
┌──────────────────────┐     ┌─────────────┐
│         App           │     │   Backend    │
│  Next.js (SPA + Auth) │────▶│   FastAPI    │
│  :3000                │     │  :8000       │
└──────────────────────┘     └─────────────┘
                                     │
                              ┌──────┴──────┐
                              │  MCP Server  │
                              │  :8080       │
                              └─────────────┘
```

| Service | Port | Description |
|---------|------|-------------|
| App | 3000 | Next.js — React SPA + auth API (JWT, OAuth, credentials) |
| Backend | 8000 | FastAPI + wiki generation engine |
| MCP Server | 8080 | AI IDE integration |

## Documentation

Full documentation is available at **[arozumenko.github.io/wikis](https://arozumenko.github.io/wikis)**

- [Quick Start](https://arozumenko.github.io/wikis/docs/quickstart) — first-time setup and generating your first wiki
- [Self-Hosting](https://arozumenko.github.io/wikis/docs/self-hosting) — Docker Compose deployment, environment variables
- [LLM Providers](https://arozumenko.github.io/wikis/docs/llm-providers) — OpenAI, Anthropic, Ollama, Gemini, Bedrock
- [MCP Integration](https://arozumenko.github.io/wikis/docs/mcp-integration) — connect your AI IDE to Wikis

## Swagger UI

The backend auto-generates OpenAPI documentation. With the backend running, visit:

[http://localhost:8000/docs](http://localhost:8000/docs)

## License

See [LICENSE](LICENSE) for details.
