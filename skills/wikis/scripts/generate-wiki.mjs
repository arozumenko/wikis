#!/usr/bin/env node
/**
 * Generate a new wiki from a repository URL.
 *
 * Usage:
 *   node generate-wiki.mjs <repo_url> [branch]
 *
 * Env:
 *   WIKIS_URL    - Base URL of the Wikis instance (required)
 *   WIKIS_TOKEN  - Bearer token for authenticated instances (optional)
 *
 * Generation is async. The response contains an invocation_id.
 * Track progress at: GET $WIKIS_URL/api/v1/invocations/<invocation_id>
 */

import { wikisUrl, headers, die } from "./_common.mjs";

const [repoUrl, branch = "main"] = process.argv.slice(2);
if (!repoUrl) die("Usage: node generate-wiki.mjs <repo_url> [branch]");

const res = await fetch(`${wikisUrl}/api/v1/generate`, {
  method: "POST",
  headers: { ...headers, "Content-Type": "application/json" },
  body: JSON.stringify({ repo_url: repoUrl, branch }),
});

if (!res.ok) {
  die(`POST /api/v1/generate failed: ${res.status} ${res.statusText}`, await res.text());
}

const data = await res.json();
console.log(JSON.stringify(data, null, 2));
console.error(`\nTrack progress: GET ${wikisUrl}/api/v1/invocations/${data.invocation_id}`);
