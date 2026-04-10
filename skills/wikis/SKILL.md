---
name: wikis
description: >
  Browse, query, and generate AI-powered code wikis via a running Wikis
  instance. Use when you need to: read architecture documentation for a
  repository, ask questions about a codebase, run deep multi-step research
  on a repo, or generate a new wiki from a repository URL. Requires
  WIKIS_URL env var pointing to the Wikis app (web: :3000 or backend: :8000).
license: Apache-2.0
compatibility: >
  Requires WIKIS_URL env var (e.g. http://localhost:3000).
  Optional WIKIS_TOKEN for authenticated instances.
  Node.js 18+ for helper scripts.
metadata:
  author: arozumenko
  version: "0.1.0"
  homepage: https://github.com/arozumenko/wikis
allowed-tools: Bash(node:) Read
---

# Wikis Skill

Interact with the [Wikis](https://github.com/arozumenko/wikis) AI documentation platform directly from your agent. All operations go through the Wikis REST API.

## Setup

Set the environment variable before running any operation:

```bash
export WIKIS_URL=http://localhost:3000   # web app (default for Docker)
# or
export WIKIS_URL=http://localhost:8000   # backend directly

# If the instance requires authentication:
export WIKIS_TOKEN=<your-jwt-token>
```

The helper scripts live in `scripts/` relative to this skill. Call them with `node`:

```bash
node skills/wikis/scripts/list-wikis.mjs
```

---

## Operations

### 1. List available wikis

```bash
node skills/wikis/scripts/list-wikis.mjs
```

Returns an array of wiki summaries — `id`, `title`, `repo_url`, `status`, `visibility`.

Use the `id` field from the response as `<wiki_id>` in subsequent operations.

---

### 2. Read a wiki and its pages

Get wiki metadata and table of contents:

```bash
node skills/wikis/scripts/get-wiki.mjs <wiki_id>
```

Get a specific page (use the `id` from the pages list):

```bash
node skills/wikis/scripts/get-wiki.mjs <wiki_id> <page_id>
```

Page content is returned as Markdown. Read it to understand architecture, data flows, module responsibilities, and design decisions.

---

### 3. Ask a question (Q&A)

Ask a focused question about a specific wiki:

```bash
node skills/wikis/scripts/ask-wiki.mjs <wiki_id> "How does authentication work?"
```

Returns a grounded answer with source citations. Best for factual, targeted questions about the codebase.

---

### 4. Deep research

Run multi-step research across a wiki's full knowledge base:

```bash
node skills/wikis/scripts/research-wiki.mjs <wiki_id> "What are the scalability bottlenecks in the pipeline?"
```

Takes longer than `ask` but synthesises information from multiple sources. Use for open-ended or cross-cutting questions.

---

### 5. Generate a new wiki

Trigger wiki generation for a repository:

```bash
node skills/wikis/scripts/generate-wiki.mjs <repo_url> [branch]
```

Examples:

```bash
node skills/wikis/scripts/generate-wiki.mjs https://github.com/org/repo main
node skills/wikis/scripts/generate-wiki.mjs https://github.com/org/repo   # defaults to main
```

Returns an `invocation_id`. Generation is async — track progress at:

```
GET $WIKIS_URL/api/v1/invocations/<invocation_id>
```

Or open `$WIKIS_URL` in a browser to watch real-time progress via the UI.

---

## Workflow example

```bash
# 1. Find the wiki for the repo you're working in
node skills/wikis/scripts/list-wikis.mjs | node -e "
  const d = JSON.parse(require('fs').readFileSync('/dev/stdin','utf8'));
  console.log(d.wikis.map(w => w.id + '  ' + w.repo_url).join('\n'));
"

# 2. Read the overview page for architectural context
node skills/wikis/scripts/get-wiki.mjs abc123

# 3. Ask a targeted question
node skills/wikis/scripts/ask-wiki.mjs abc123 "Which service owns database migrations?"

# 4. Deep research for a complex topic
node skills/wikis/scripts/research-wiki.mjs abc123 "Trace the full request lifecycle for wiki generation"
```

## Error handling

All scripts exit with code `0` on success and `1` on error, printing a JSON error object to stderr. If `WIKIS_URL` is not set, the script aborts immediately with a clear message.
