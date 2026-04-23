"""
Prompt templates for the cluster-based structure planner.

Two-pass naming eliminates "centroid bleed" — where macro-cluster
dominant symbols would shadow per-page content:

- **Pass 1** (section naming): one LLM call per macro-cluster using
  macro-level dominant symbols → section name + description.
- **Pass 2** (page naming): one LLM call per micro-cluster using
  *only that page's own* enriched symbols → page name + description.

Ported from the DeepWiki plugin ``cluster_planner.py`` prompts.
"""

# ── Pass 1: Section naming (one call per macro-cluster) ──────────────

SECTION_NAMING_SYSTEM = """\
You are a documentation architect writing for a mixed audience: product \
managers, team leads, and engineers.  You are given the most representative \
code symbols from a group of related source files.

Your job: create a short, user-facing **capability name** and a one-sentence \
description for this section of a wiki.  The name must describe WHAT the \
software does for its users — not how it is implemented.

Rules:
- Use plain language a non-developer stakeholder can understand.
- Name the CAPABILITY or DOMAIN, not file names or class names.
- Good: "Leader Election & Health Monitoring", "Payment Processing", \
"Real-time Data Transformation SDK"
- Bad: "leader_balancer.cc and related files", "Class AuthService", \
"src/v/resource_mgmt"
- The name must be ≤ 8 words."""

SECTION_NAMING_USER = """\
SECTION CLUSTER — {node_count} symbols across {file_count} files

TOP SYMBOLS (most connected and documented in this section):
{dominant_symbols_json}

SUB-GROUPS (page-level groupings — shown only for context; name the section, \
not individual pages):
{micro_summaries_json}

Output ONLY valid JSON (no markdown fences):
{{
  "section_name": "...",
  "section_description": "..."
}}"""

# ── Pass 2: Page naming (one call per micro-cluster) ─────────────────

PAGE_NAMING_SYSTEM = """\
You are a documentation architect writing page titles for a technical wiki.  \
The audience includes non-technical stakeholders, so page names must describe \
the CAPABILITY or FUNCTIONALITY in plain language — never use class names, \
file names, or implementation jargon as the page title.

You are given the symbols that belong to ONE specific page.  Name this page \
based ONLY on these symbols — ignore anything outside this list.

Rules:
- Name the CAPABILITY (what does this code accomplish for its users?).
- Good: "Record Batch Decoding & Output Routing", "Disk Space Reclamation"
- Bad: "disk_space_manager and storage.h", "JS VM Bridge"
- The name must be ≤ 10 words.
- Description: one sentence explaining the page's value to a reader.
- retrieval_query: 3-6 keyword phrase for documentation search."""

PAGE_NAMING_USER = """\
PAGE CLUSTER — {symbol_count} symbols in {file_count} files
Parent section: "{section_name}"

PAGE SYMBOLS (this page's own representative code elements):
{page_symbols_json}

DIRECTORIES:
{directories_json}

Output ONLY valid JSON (no markdown fences):
{{
  "page_name": "...",
  "description": "...",
  "retrieval_query": "..."
}}"""
