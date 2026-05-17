/**
 * @jest-environment jsdom
 */
import { act, renderHook } from '@testing-library/react';
import { useThemeMode } from '../useThemeMode';

const STORAGE_KEY = 'wikis-theme-mode';

function mockMatchMedia(matches: boolean) {
  // jsdom doesn't implement matchMedia; provide a minimal stub.
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    configurable: true,
    value: jest.fn().mockImplementation((query: string) => ({
      matches,
      media: query,
      onchange: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      addListener: jest.fn(),
      removeListener: jest.fn(),
      dispatchEvent: jest.fn(),
    })),
  });
}

describe('useThemeMode', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('renders exactly twice: pre-effect (mounted=false) then post-effect (mounted=true)', () => {
    // Rio's review caught that the previous test only observed the
    // post-effect state. To actually verify the SSR-guard contract
    // we need to capture both renders. A counter accumulates the
    // hook output per render so we can inspect the pre-effect
    // snapshot directly.
    mockMatchMedia(true);
    const captured: Array<{ mode: string; mounted: boolean }> = [];

    renderHook(() => {
      const r = useThemeMode();
      captured.push({ mode: r.mode, mounted: r.mounted });
      return r;
    });

    // First render is the synchronous useState init (pre-effect)
    // and must report mounted=false — that's the gate SSR consumers
    // rely on. The post-effect render then flips mounted to true.
    expect(captured.length).toBeGreaterThanOrEqual(2);
    expect(captured[0].mounted).toBe(false);
    expect(captured[captured.length - 1].mounted).toBe(true);
  });

  it('mount effect produces a single re-render (batched mode + mounted)', () => {
    // Rio's review: setMode + setMounted as two calls would only
    // batch on React 18; combining into one setState guarantees
    // a single re-render across React versions and microtask
    // boundaries. Counter pins the invariant.
    mockMatchMedia(true);
    let renderCount = 0;

    renderHook(() => {
      renderCount += 1;
      return useThemeMode();
    });

    // 1 synchronous initial render + 1 post-effect render = 2.
    // A third render would indicate the mount effect split into
    // two state updates that didn't batch.
    expect(renderCount).toBe(2);
  });

  it('reads system preference when no localStorage value is set', () => {
    mockMatchMedia(true); // prefers dark
    const { result } = renderHook(() => useThemeMode());
    expect(result.current.mode).toBe('dark');
  });

  it('prefers saved mode over system preference', () => {
    window.localStorage.setItem(STORAGE_KEY, 'light');
    mockMatchMedia(true); // system is dark
    const { result } = renderHook(() => useThemeMode());
    expect(result.current.mode).toBe('light');
  });

  it('toggleMode persists to localStorage and flips the mode', () => {
    mockMatchMedia(false);
    const { result } = renderHook(() => useThemeMode());
    expect(result.current.mode).toBe('light');

    act(() => {
      result.current.toggleMode();
    });

    expect(result.current.mode).toBe('dark');
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('dark');
  });
});
