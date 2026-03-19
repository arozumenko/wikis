const API_BASE = ''; // Same origin — all API calls proxied through Next.js
const AUTH_URL = ''; // Same origin — auth is in the same process

export class ApiError extends Error {
  constructor(
    public status: number,
    public body: unknown,
  ) {
    super(`API error ${status}`);
    this.name = 'ApiError';
  }
}

let cachedToken: string | null = null;
let tokenExpiry = 0;
let redirecting401 = false;

// Reset the redirect guard if navigation was interrupted (e.g. user hit back).
if (typeof document !== 'undefined') {
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      redirecting401 = false;
    }
  });
}

export async function getAuthToken(): Promise<string | null> {
  if (cachedToken && Date.now() < tokenExpiry) {
    return cachedToken;
  }

  try {
    const resp = await fetch(`${AUTH_URL}/api/auth/token`, {
      credentials: 'include',
    });
    if (!resp.ok) return null;
    const data = await resp.json();
    cachedToken = data.token;
    // Cache for 55 minutes (tokens last 24h, refresh well before expiry)
    tokenExpiry = Date.now() + 55 * 60 * 1000;
    return cachedToken;
  } catch {
    return null;
  }
}

export function clearTokenCache(): void {
  cachedToken = null;
  tokenExpiry = 0;
}

export async function apiRequest<T>(path: string, options?: RequestInit): Promise<T> {
  const token = await getAuthToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
    ...((options?.headers as Record<string, string>) ?? {}),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const resp = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });

  if (!resp.ok) {
    if (resp.status === 401) {
      if (redirecting401) return new Promise<never>(() => {});
      redirecting401 = true;
      clearTokenCache();
      const returnPath = window.location.pathname + window.location.search;
      // callbackUrl signals the middleware to clear any stale session cookie
      window.location.href = `/login?callbackUrl=${encodeURIComponent(returnPath)}`;
      // Never resolves — page is navigating away
      return new Promise<never>(() => {});
    }
    const body = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new ApiError(resp.status, body);
  }

  return resp.json();
}
