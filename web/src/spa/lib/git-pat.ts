/**
 * Git PAT storage helpers.
 *
 * Each PAT is keyed by a short hash of the repo URL so that multiple repos
 * can be stored independently. No OAuth — just save-and-retrieve.
 *
 * Storage key format: `wikis.connections.git.{8-hex-chars}`
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface GitConnectionEntry {
  provider: 'git';
  id: string;         // first 8 hex chars of SHA-256(repo_url)
  repo_url: string;
  branch: string;
  pat: string;
  label: string;
  created_at: number; // epoch ms
}

// ---------------------------------------------------------------------------
// Storage key helpers
// ---------------------------------------------------------------------------

const KEY_PREFIX = 'wikis.connections.git.';

/**
 * Compute a stable 8-hex-char identifier for a repo URL.
 * Uses Web Crypto API — browser-native, no library required.
 */
export async function repoUrlId(repoUrl: string): Promise<string> {
  const encoded = new TextEncoder().encode(repoUrl);
  const digest = await crypto.subtle.digest('SHA-256', encoded);
  const hex = Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
  return hex.slice(0, 8);
}

function storageKey(id: string): string {
  return `${KEY_PREFIX}${id}`;
}

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

/** Persist a Git PAT connection. Overwrites any prior entry for the same repo_url id. */
export function saveGitPAT(entry: GitConnectionEntry): void {
  try {
    localStorage.setItem(storageKey(entry.id), JSON.stringify(entry));
  } catch {
    // Storage quota exceeded or private browsing — fail silently
  }
}

/** Remove a Git PAT connection by its id. */
export function removeGitPAT(id: string): void {
  try {
    localStorage.removeItem(storageKey(id));
  } catch {
    // ignore
  }
}

/** Load all stored Git PAT connections. */
export function loadAllGitPATs(): GitConnectionEntry[] {
  const results: GitConnectionEntry[] = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (!key || !key.startsWith(KEY_PREFIX)) continue;
      const raw = localStorage.getItem(key);
      if (!raw) continue;
      try {
        const entry = JSON.parse(raw) as GitConnectionEntry;
        if (entry.provider === 'git') results.push(entry);
      } catch {
        // Corrupt entry — skip
      }
    }
  } catch {
    // localStorage unavailable
  }
  return results.sort((a, b) => a.created_at - b.created_at);
}

/** Load a single Git PAT connection by id, or null if not found. */
export function loadGitPAT(id: string): GitConnectionEntry | null {
  try {
    const raw = localStorage.getItem(storageKey(id));
    if (!raw) return null;
    const entry = JSON.parse(raw) as GitConnectionEntry;
    if (entry.provider === 'git') return entry;
  } catch {
    // ignore
  }
  return null;
}
