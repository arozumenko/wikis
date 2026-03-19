import { useCallback, useEffect, useRef, useState } from 'react';
import { clearTokenCache } from '../api/client';

const AUTH_URL = ''; // Same origin

interface User {
  id: string;
  name: string;
  email: string;
  image?: string | null;
}

interface AuthState {
  user: User | null;
  loading: boolean;
  authenticated: boolean;
}

export function useAuth() {
  const [state, setState] = useState<AuthState>({
    user: null,
    loading: true,
    authenticated: false,
  });
  const redirectingRef = useRef(false);

  useEffect(() => {
    let cancelled = false;

    async function checkSession() {
      try {
        const resp = await fetch(`${AUTH_URL}/api/auth/get-session`, {
          credentials: 'include',
        });

        if (!resp.ok) {
          if (!cancelled) setState({ user: null, loading: false, authenticated: false });
          return;
        }

        const data = await resp.json();

        if (!cancelled) {
          // Better Auth returns { session: {...}, user: {...} }
          const user = data?.user;
          if (user) {
            setState({
              user: {
                id: user.id,
                name: user.name,
                email: user.email,
                image: user.image ?? null,
              },
              loading: false,
              authenticated: true,
            });
          } else {
            // Session cookie exists but session is expired/invalid.
            // Sign out to clear the stale cookie before redirecting to /login,
            // otherwise middleware sees the cookie and redirects back here (loop).
            await fetch(`${AUTH_URL}/api/auth/sign-out`, {
              method: 'POST',
              credentials: 'include',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({}),
            }).catch(() => {});
            setState({ user: null, loading: false, authenticated: false });
          }
        }
      } catch {
        if (!cancelled) {
          setState({ user: null, loading: false, authenticated: false });
        }
      }
    }

    checkSession();
    return () => {
      cancelled = true;
    };
  }, []);

  const signIn = useCallback(() => {
    // Guard against multiple calls (Safari remounts can re-trigger effects)
    if (redirectingRef.current) return;
    redirectingRef.current = true;

    clearTokenCache();
    const returnPath = window.location.pathname + window.location.search;
    // callbackUrl signals the middleware to clear any stale session cookie
    // and let the request through to /login — no async sign-out needed.
    window.location.href = `/login?callbackUrl=${encodeURIComponent(returnPath)}`;
  }, []);

  const signOut = useCallback(async () => {
    if (redirectingRef.current) return;
    redirectingRef.current = true;

    clearTokenCache();
    try {
      await fetch(`${AUTH_URL}/api/auth/sign-out`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
    } catch {
      // Continue with redirect even if API call fails
    }
    window.location.href = `/login`;
  }, []);

  return { ...state, signIn, signOut };
}
