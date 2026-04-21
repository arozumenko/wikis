#!/usr/bin/env node
/**
 * Run deep multi-step research on a wiki.
 *
 * Usage:
 *   node research-wiki.mjs <wiki_id> "<question>"
 *
 * Env:
 *   WIKIS_URL    - Base URL of the Wikis instance (required)
 *   WIKIS_TOKEN  - Bearer token for authenticated instances (optional)
 */

import { wikisUrl, headers, die } from "./_common.mjs";

const [wikiId, question] = process.argv.slice(2);
if (!wikiId || !question) die("Usage: node research-wiki.mjs <wiki_id> \"<question>\"");

const res = await fetch(`${wikisUrl}/api/v1/research`, {
  method: "POST",
  headers: { ...headers, "Content-Type": "application/json" },
  body: JSON.stringify({ wiki_id: wikiId, question }),
});

if (!res.ok) {
  die(`POST /api/v1/research failed: ${res.status} ${res.statusText}`, await res.text());
}

console.log(JSON.stringify(await res.json(), null, 2));
