#!/usr/bin/env node
/**
 * Get a wiki by ID, or a specific page within it.
 *
 * Usage:
 *   node get-wiki.mjs <wiki_id>
 *   node get-wiki.mjs <wiki_id> <page_id>
 *
 * Env:
 *   WIKIS_URL    - Base URL of the Wikis instance (required)
 *   WIKIS_TOKEN  - Bearer token for authenticated instances (optional)
 */

import { wikisUrl, headers, die } from "./_common.mjs";

const [wikiId, pageId] = process.argv.slice(2);
if (!wikiId) die("Usage: node get-wiki.mjs <wiki_id> [page_id]");

const path = pageId
  ? `/api/v1/wikis/${wikiId}/pages/${pageId}`
  : `/api/v1/wikis/${wikiId}`;

const res = await fetch(`${wikisUrl}${path}`, { headers });

if (!res.ok) {
  die(`GET ${path} failed: ${res.status} ${res.statusText}`, await res.text());
}

const data = await res.json();
console.log(JSON.stringify(data, null, 2));
