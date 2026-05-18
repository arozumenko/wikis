/**
 * @jest-environment jsdom
 */
import { act, renderHook } from '@testing-library/react';
import { useConnections } from '../useConnections';
import type { AtlassianConnection } from '../useConnections';
import type { GitConnectionEntry } from '../../lib/git-pat';

// Mock atlassian-oauth so tests never hit the network
jest.mock('../../lib/atlassian-oauth', () => ({
  refreshAtlassianTokens: jest.fn(),
  // Preserve other exports in case they're needed transitively
  fetchAccessibleResources: jest.fn(),
  startAtlassianOAuth: jest.fn(),
  completeAtlassianOAuth: jest.fn(),
}));

import { refreshAtlassianTokens } from '../../lib/atlassian-oauth';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeAtlassianConnection(overrides?: Partial<AtlassianConnection>): AtlassianConnection {
  return {
    provider: 'atlassian',
    access_token: 'access-tok',
    refresh_token: 'refresh-tok',
    expires_at: Date.now() + 3600 * 1000, // 1 hour from now
    cloud_id: 'cloud-123',
    site_name: 'test.atlassian.net',
    accessible_resources: [
      {
        id: 'cloud-123',
        name: 'test.atlassian.net',
        url: 'https://test.atlassian.net',
        scopes: ['read:confluence-content.all'],
      },
    ],
    created_at: Date.now(),
    ...overrides,
  };
}

function makeGitConnection(overrides?: Partial<GitConnectionEntry>): GitConnectionEntry {
  return {
    provider: 'git',
    id: 'abcd1234',
    repo_url: 'https://github.com/org/repo',
    branch: 'main',
    pat: 'ghp_token',
    label: 'My repo',
    created_at: Date.now(),
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('useConnections', () => {
  beforeEach(() => {
    window.localStorage.clear();
    jest.clearAllMocks();
  });

  // -------------------------------------------------------------------------
  // Initial state
  // -------------------------------------------------------------------------

  it('starts with no connections', () => {
    const { result } = renderHook(() => useConnections());
    expect(result.current.connections).toHaveLength(0);
    expect(result.current.atlassian).toBeNull();
  });

  // -------------------------------------------------------------------------
  // Atlassian: save + remove
  // -------------------------------------------------------------------------

  it('saveAtlassian persists to localStorage and updates state', () => {
    const { result } = renderHook(() => useConnections());
    const conn = makeAtlassianConnection();

    act(() => {
      result.current.saveAtlassian(conn);
    });

    expect(result.current.atlassian).not.toBeNull();
    expect(result.current.atlassian?.cloud_id).toBe('cloud-123');
    expect(result.current.connections).toHaveLength(1);

    const stored = JSON.parse(localStorage.getItem('wikis.connections.atlassian')!);
    expect(stored.cloud_id).toBe('cloud-123');
  });

  it('removeAtlassian clears state and localStorage', () => {
    const { result } = renderHook(() => useConnections());
    const conn = makeAtlassianConnection();

    act(() => {
      result.current.saveAtlassian(conn);
    });

    act(() => {
      result.current.removeAtlassian();
    });

    expect(result.current.atlassian).toBeNull();
    expect(result.current.connections).toHaveLength(0);
    expect(localStorage.getItem('wikis.connections.atlassian')).toBeNull();
  });

  // -------------------------------------------------------------------------
  // Git: save + remove
  // -------------------------------------------------------------------------

  it('saveGitConnection adds a connection', () => {
    const { result } = renderHook(() => useConnections());
    const conn = makeGitConnection();

    act(() => {
      result.current.saveGitConnection(conn);
    });

    expect(result.current.connections).toHaveLength(1);
    const stored = localStorage.getItem('wikis.connections.git.abcd1234');
    expect(stored).not.toBeNull();
  });

  it('removeGitConnection removes by id', () => {
    const { result } = renderHook(() => useConnections());
    const conn = makeGitConnection();

    act(() => {
      result.current.saveGitConnection(conn);
    });

    act(() => {
      result.current.removeGitConnection('abcd1234');
    });

    expect(result.current.connections).toHaveLength(0);
    expect(localStorage.getItem('wikis.connections.git.abcd1234')).toBeNull();
  });

  it('multiple git connections coexist', () => {
    const { result } = renderHook(() => useConnections());
    const a = makeGitConnection({ id: 'aaaaaaaa', repo_url: 'https://github.com/org/a' });
    const b = makeGitConnection({ id: 'bbbbbbbb', repo_url: 'https://github.com/org/b' });

    act(() => {
      result.current.saveGitConnection(a);
      result.current.saveGitConnection(b);
    });

    expect(result.current.connections).toHaveLength(2);
  });

  // -------------------------------------------------------------------------
  // Mixed: atlassian + git appear together
  // -------------------------------------------------------------------------

  it('connections list contains both Atlassian and Git entries', () => {
    const { result } = renderHook(() => useConnections());

    act(() => {
      result.current.saveAtlassian(makeAtlassianConnection());
      result.current.saveGitConnection(makeGitConnection());
    });

    expect(result.current.connections).toHaveLength(2);
    const providers = result.current.connections.map((c) => c.provider);
    expect(providers).toContain('atlassian');
    expect(providers).toContain('git');
  });

  // -------------------------------------------------------------------------
  // refreshAtlassianIfNeeded
  // -------------------------------------------------------------------------

  it('refreshAtlassianIfNeeded returns null when no connection is stored', async () => {
    const { result } = renderHook(() => useConnections());
    const updated = await result.current.refreshAtlassianIfNeeded();
    expect(updated).toBeNull();
  });

  it('refreshAtlassianIfNeeded returns current connection when not expiring soon', async () => {
    const { result } = renderHook(() => useConnections());
    const conn = makeAtlassianConnection({
      expires_at: Date.now() + 30 * 60 * 1000, // 30 minutes — well within window
    });

    act(() => {
      result.current.saveAtlassian(conn);
    });

    // No network call expected — token is fresh
    const updated = await result.current.refreshAtlassianIfNeeded();
    expect(updated?.access_token).toBe('access-tok');
  });

  // -------------------------------------------------------------------------
  // refreshAtlassianIfNeeded — within-5-min-of-expiry branch
  // -------------------------------------------------------------------------

  it('refreshAtlassianIfNeeded happy path: rotates tokens when expiring soon', async () => {
    const mockRefresh = refreshAtlassianTokens as jest.MockedFunction<typeof refreshAtlassianTokens>;
    const rotatedAt = Date.now() + 3600 * 1000;
    mockRefresh.mockResolvedValueOnce({
      access_token: 'rotated-access',
      refresh_token: 'rotated-refresh',
      expires_at: rotatedAt,
      scope: 'read:confluence-content.all',
    });

    const { result } = renderHook(() => useConnections());
    // expires_at 4 minutes from now — inside the 5-minute refresh window
    const conn = makeAtlassianConnection({ expires_at: Date.now() + 4 * 60 * 1000 });

    act(() => {
      result.current.saveAtlassian(conn);
    });

    let updated: AtlassianConnection | null = null;
    await act(async () => {
      updated = await result.current.refreshAtlassianIfNeeded();
    });

    // refreshAtlassianTokens was called with the old refresh token
    expect(mockRefresh).toHaveBeenCalledTimes(1);
    expect(mockRefresh).toHaveBeenCalledWith('refresh-tok');

    // Returned connection has rotated tokens
    expect(updated).not.toBeNull();
    expect(updated!.access_token).toBe('rotated-access');
    expect(updated!.refresh_token).toBe('rotated-refresh');
    expect(updated!.expires_at).toBe(rotatedAt);

    // Hook state updated
    expect(result.current.atlassian?.access_token).toBe('rotated-access');

    // localStorage updated
    const stored = JSON.parse(localStorage.getItem('wikis.connections.atlassian')!);
    expect(stored.access_token).toBe('rotated-access');
    expect(stored.refresh_token).toBe('rotated-refresh');
    // scope is not persisted
    expect(stored.scope).toBeUndefined();
  });

  it('refreshAtlassianIfNeeded failure path: localStorage and state unchanged when refresh rejects', async () => {
    const mockRefresh = refreshAtlassianTokens as jest.MockedFunction<typeof refreshAtlassianTokens>;
    mockRefresh.mockRejectedValueOnce(new Error('Network error'));

    const { result } = renderHook(() => useConnections());
    const conn = makeAtlassianConnection({ expires_at: Date.now() + 4 * 60 * 1000 });

    act(() => {
      result.current.saveAtlassian(conn);
    });

    // The rejection should propagate out of the hook
    await expect(
      act(async () => {
        await result.current.refreshAtlassianIfNeeded();
      }),
    ).rejects.toThrow('Network error');

    // State unchanged — still has the original tokens
    expect(result.current.atlassian?.access_token).toBe('access-tok');

    // localStorage unchanged
    const stored = JSON.parse(localStorage.getItem('wikis.connections.atlassian')!);
    expect(stored.access_token).toBe('access-tok');
  });

  // -------------------------------------------------------------------------
  // refreshAtlassianNow — force-refresh regardless of expiry
  // -------------------------------------------------------------------------

  it('refreshAtlassianNow forces a refresh even when token is fresh', async () => {
    const mockRefresh = refreshAtlassianTokens as jest.MockedFunction<typeof refreshAtlassianTokens>;
    const rotatedAt = Date.now() + 7200 * 1000;
    mockRefresh.mockResolvedValueOnce({
      access_token: 'force-rotated-access',
      refresh_token: 'force-rotated-refresh',
      expires_at: rotatedAt,
      scope: 'read:confluence-content.all',
    });

    const { result } = renderHook(() => useConnections());
    // Token is fresh — 1 hour from now (outside the 5-min window)
    const conn = makeAtlassianConnection({ expires_at: Date.now() + 3600 * 1000 });

    act(() => {
      result.current.saveAtlassian(conn);
    });

    let updated: AtlassianConnection | null = null;
    await act(async () => {
      updated = await result.current.refreshAtlassianNow();
    });

    expect(mockRefresh).toHaveBeenCalledTimes(1);
    expect(updated!.access_token).toBe('force-rotated-access');
    expect(result.current.atlassian?.access_token).toBe('force-rotated-access');
  });

  it('refreshAtlassianNow returns null when no connection is stored', async () => {
    const { result } = renderHook(() => useConnections());
    const updated = await result.current.refreshAtlassianNow();
    expect(updated).toBeNull();
  });
});
