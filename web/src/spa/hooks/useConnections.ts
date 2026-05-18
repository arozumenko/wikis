/**
 * useConnections — localStorage-backed hook for managing stored connections
 * (Atlassian OAuth + Git PATs).
 *
 * Reactive: uses `window.addEventListener('storage')` so the hook syncs
 * across browser tabs.
 */
import { useCallback, useEffect, useState } from 'react';
import type { AccessibleResource } from '../lib/atlassian-oauth';
import { refreshAtlassianTokens } from '../lib/atlassian-oauth';
import {
  loadAllGitPATs,
  removeGitPAT,
  saveGitPAT,
  type GitConnectionEntry,
} from '../lib/git-pat';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AtlassianConnection {
  provider: 'atlassian';
  access_token: string;
  refresh_token: string;
  /** Epoch milliseconds */
  expires_at: number;
  cloud_id: string;
  site_name: string;
  accessible_resources: AccessibleResource[];
  /** Epoch milliseconds */
  created_at: number;
}

// Re-export GitConnection so callers can import from one place
export type { GitConnectionEntry as GitConnection };

export type Connection = AtlassianConnection | GitConnectionEntry;

export interface UseConnectionsResult {
  connections: Connection[];
  atlassian: AtlassianConnection | null;
  saveAtlassian: (c: AtlassianConnection) => void;
  removeAtlassian: () => void;
  saveGitConnection: (c: GitConnectionEntry) => void;
  removeGitConnection: (id: string) => void;
  /**
   * If the Atlassian token expires within 5 minutes, calls the refresh
   * endpoint, stores the rotated tokens, and returns the updated connection.
   * Otherwise returns the current connection unchanged.
   *
   * Returns null if there is no Atlassian connection.
   */
  refreshAtlassianIfNeeded: () => Promise<AtlassianConnection | null>;
  /**
   * Unconditionally refreshes the Atlassian access token regardless of expiry.
   * Useful for the manual "Refresh" button in the UI.
   *
   * Returns null if there is no Atlassian connection.
   */
  refreshAtlassianNow: () => Promise<AtlassianConnection | null>;
}

// ---------------------------------------------------------------------------
// Storage key for the single Atlassian identity
// ---------------------------------------------------------------------------

const ATLASSIAN_KEY = 'wikis.connections.atlassian';

// Refresh window: 5 minutes before expiry
const REFRESH_WINDOW_MS = 5 * 60 * 1000;

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

function loadAtlassian(): AtlassianConnection | null {
  try {
    const raw = localStorage.getItem(ATLASSIAN_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as AtlassianConnection;
    if (parsed.provider === 'atlassian') return parsed;
  } catch {
    // Corrupt entry
  }
  return null;
}

function persistAtlassian(c: AtlassianConnection): void {
  try {
    localStorage.setItem(ATLASSIAN_KEY, JSON.stringify(c));
  } catch {
    // Storage quota exceeded — fail silently
  }
}

function clearAtlassian(): void {
  try {
    localStorage.removeItem(ATLASSIAN_KEY);
  } catch {
    // ignore
  }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useConnections(): UseConnectionsResult {
  const [atlassian, setAtlassian] = useState<AtlassianConnection | null>(() =>
    typeof window !== 'undefined' ? loadAtlassian() : null,
  );
  const [gitConnections, setGitConnections] = useState<GitConnectionEntry[]>(() =>
    typeof window !== 'undefined' ? loadAllGitPATs() : [],
  );

  // Sync across tabs via the storage event
  useEffect(() => {
    function onStorage(e: StorageEvent) {
      if (e.key === ATLASSIAN_KEY || e.key === null) {
        setAtlassian(loadAtlassian());
      }
      if (e.key === null || (e.key && e.key.startsWith('wikis.connections.git.'))) {
        setGitConnections(loadAllGitPATs());
      }
    }

    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  // -------------------------------------------------------------------------
  // Atlassian
  // -------------------------------------------------------------------------

  const saveAtlassian = useCallback((c: AtlassianConnection) => {
    persistAtlassian(c);
    setAtlassian(c);
  }, []);

  const removeAtlassian = useCallback(() => {
    clearAtlassian();
    setAtlassian(null);
  }, []);

  /** Internal: call the refresh endpoint and persist + update state. */
  const doRefresh = useCallback(async (current: AtlassianConnection): Promise<AtlassianConnection> => {
    const tokens = await refreshAtlassianTokens(current.refresh_token);
    const updated: AtlassianConnection = {
      ...current,
      access_token: tokens.access_token,
      refresh_token: tokens.refresh_token,
      expires_at: tokens.expires_at,
      // scope is not part of AtlassianConnection — not persisted
    };
    persistAtlassian(updated);
    setAtlassian(updated);
    return updated;
  }, []);

  const refreshAtlassianIfNeeded = useCallback(async (): Promise<AtlassianConnection | null> => {
    const current = loadAtlassian();
    if (!current) return null;

    const needsRefresh = current.expires_at - Date.now() < REFRESH_WINDOW_MS;
    if (!needsRefresh) return current;

    return doRefresh(current);
  }, [doRefresh]);

  const refreshAtlassianNow = useCallback(async (): Promise<AtlassianConnection | null> => {
    const current = loadAtlassian();
    if (!current) return null;
    return doRefresh(current);
  }, [doRefresh]);

  // -------------------------------------------------------------------------
  // Git PATs
  // -------------------------------------------------------------------------

  const saveGitConnection = useCallback((c: GitConnectionEntry) => {
    saveGitPAT(c);
    setGitConnections(loadAllGitPATs());
  }, []);

  const removeGitConnection = useCallback((id: string) => {
    removeGitPAT(id);
    setGitConnections(loadAllGitPATs());
  }, []);

  // -------------------------------------------------------------------------
  // Aggregated list
  // -------------------------------------------------------------------------

  const connections: Connection[] = [
    ...(atlassian ? [atlassian] : []),
    ...gitConnections,
  ];

  return {
    connections,
    atlassian,
    saveAtlassian,
    removeAtlassian,
    saveGitConnection,
    removeGitConnection,
    refreshAtlassianIfNeeded,
    refreshAtlassianNow,
  };
}
