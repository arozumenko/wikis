#!/usr/bin/env node
/**
 * List all available wikis.
 *
 * Usage:
 *   node list-wikis.mjs
 *
 * Env:
 *   WIKIS_URL    - Base URL of the Wikis instance (required)
 *   WIKIS_TOKEN  - Bearer token for authenticated instances (optional)
 */

import { wikisUrl, headers, die } from "./_common.mjs";

const res = await fetch(`${wikisUrl}/api/v1/wikis`, { headers });

if (!res.ok) {
  die(`GET /api/v1/wikis failed: ${res.status} ${res.statusText}`, await res.text());
}

console.log(JSON.stringify(await res.json(), null, 2));
