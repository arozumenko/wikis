import { useCallback, useEffect, useState } from 'react';

type ThemeMode = 'light' | 'dark';

const STORAGE_KEY = 'wikis-theme-mode';

function getSystemPreference(): ThemeMode {
  if (typeof window === 'undefined') return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function getSavedMode(): ThemeMode | null {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === 'light' || saved === 'dark') return saved;
  } catch {
    // localStorage unavailable
  }
  return null;
}

interface ThemeState {
  mode: ThemeMode;
  /**
   * ``mounted`` flips to ``true`` after the first client effect.
   * SSR-rendered consumers (the Next.js login page) gate their
   * themed UI on this flag to avoid a hydration mismatch: the SSR
   * ``useState`` initializer returns ``'light'`` (no ``window`` on
   * the server) while the client init returns the real preference,
   * and that mismatch leaves MUI's emotion classes in a mixed
   * light/dark state — the white-on-white input symptom. Pure
   * client-rendered SPA consumers (``ssr: false``) can ignore this
   * flag because there is no SSR step to disagree with.
   */
  mounted: boolean;
}

export function useThemeMode() {
  // Single state object so the mount effect dispatches *one* update
  // (mode + mounted together) — guaranteed to produce a single
  // re-render regardless of whether React's auto-batching is in
  // effect. Two separate ``setState`` calls would batch on React
  // 18 but split on older versions or across microtask boundaries.
  const [state, setState] = useState<ThemeState>(() => ({
    mode: getSavedMode() ?? getSystemPreference(),
    mounted: false,
  }));
  const { mode, mounted } = state;

  // Sync with actual client preference on mount (SSR always defaults to 'light')
  useEffect(() => {
    setState({
      mode: getSavedMode() ?? getSystemPreference(),
      mounted: true,
    });
  }, []);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e: MediaQueryListEvent) => {
      if (!getSavedMode()) {
        setState((prev) => ({ ...prev, mode: e.matches ? 'dark' : 'light' }));
      }
    };
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  const toggleMode = useCallback(() => {
    setState((prev) => {
      const next = prev.mode === 'light' ? 'dark' : 'light';
      try {
        localStorage.setItem(STORAGE_KEY, next);
      } catch {
        // localStorage unavailable
      }
      return { ...prev, mode: next };
    });
  }, []);

  return { mode, toggleMode, mounted } as const;
}
