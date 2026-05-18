import { useEffect, useRef } from 'react';

/**
 * Fire ``onResume`` when the tab regains user attention or network.
 *
 * #191: the SPA was treating its cached React/SSE buffer as ground truth.
 * After laptop sleep, the cache could disagree with the backend for hours.
 * This hook is the single "ask the server again" trigger — pages and global
 * containers wire it up and refetch whatever they own.
 *
 * Fires on:
 *  - ``visibilitychange`` when the document becomes visible
 *  - ``focus`` on the window
 *  - ``online`` on the window
 *
 * A 250 ms debounce coalesces the burst (browsers commonly fire visible +
 * focus + online back-to-back on wake) so the callback runs once per wake.
 */
export function useResume(onResume: () => void, enabled: boolean = true): void {
  const callbackRef = useRef(onResume);
  callbackRef.current = onResume;

  useEffect(() => {
    if (!enabled) return;
    if (typeof document === 'undefined' || typeof window === 'undefined') return;

    let timer: ReturnType<typeof setTimeout> | null = null;
    const fire = () => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = null;
        callbackRef.current();
      }, 250);
    };

    const onVisibility = () => {
      if (document.visibilityState === 'visible') fire();
    };

    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('focus', fire);
    window.addEventListener('online', fire);
    return () => {
      if (timer) clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisibility);
      window.removeEventListener('focus', fire);
      window.removeEventListener('online', fire);
    };
  }, [enabled]);
}
