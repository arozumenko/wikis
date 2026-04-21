#!/usr/bin/env node
/**
 * Ask a question about a wiki (Q&A).
 *
 * Usage:
 *   node ask-wiki.mjs <wiki_id> "<question>"
 *
 * Env:
 *   WIKIS_URL    - Base URL of the Wikis instance (required)
 *   WIKIS_TOKEN  - Bearer token for authenticated instances (optional)
 */

import { wikisUrl, headers, die } from "./_common.mjs";

const [wikiId, question] = process.argv.slice(2);
if (!wikiId || !question) die("Usage: node ask-wiki.mjs <wiki_id> \"<question>\"");

const res = await fetch(`${wikisUrl}/api/v1/ask`, {
  method: "POST",
  headers: { ...headers, "Content-Type": "application/json" },
  body: JSON.stringify({ wiki_id: wikiId, question }),
});

if (!res.ok) {
  die(`POST /api/v1/ask failed: ${res.status} ${res.statusText}`, await res.text());
}

console.log(JSON.stringify(await res.json(), null, 2));
