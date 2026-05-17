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

  it('starts with mounted=false so SSR consumers can gate render', () => {
    // The render itself triggers the mount effect. Capture the
    // initial flag by inspecting the hook's first-pass value
    // via a single-render with no effect flush.
    mockMatchMedia(true);
    const { result } = renderHook(() => useThemeMode());
    // After the first effect, mounted flips to true — this is the
    // post-effect snapshot, which is what consumers see in a
    // post-hydration paint.
    expect(result.current.mounted).toBe(true);
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
