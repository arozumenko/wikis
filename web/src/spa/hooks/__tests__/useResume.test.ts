/**
 * @jest-environment jsdom
 *
 * #191 — useResume fires when the tab regains attention so callers can
 * re-ask the backend instead of trusting their cached state.
 */
import { act, renderHook } from '@testing-library/react';
import { useResume } from '../useResume';

describe('useResume', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });
  afterEach(() => {
    jest.useRealTimers();
  });

  function fireVisibilityVisible() {
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      get: () => 'visible',
    });
    document.dispatchEvent(new Event('visibilitychange'));
  }

  function flushDebounce() {
    act(() => {
      jest.advanceTimersByTime(250);
    });
  }

  it('fires on focus after debounce', () => {
    const cb = jest.fn();
    renderHook(() => useResume(cb));

    act(() => {
      window.dispatchEvent(new Event('focus'));
    });
    expect(cb).not.toHaveBeenCalled();
    flushDebounce();
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it('fires on online', () => {
    const cb = jest.fn();
    renderHook(() => useResume(cb));

    act(() => {
      window.dispatchEvent(new Event('online'));
    });
    flushDebounce();
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it('fires on visibilitychange when document becomes visible', () => {
    const cb = jest.fn();
    renderHook(() => useResume(cb));

    fireVisibilityVisible();
    flushDebounce();
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it('ignores visibilitychange to hidden', () => {
    const cb = jest.fn();
    renderHook(() => useResume(cb));

    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      get: () => 'hidden',
    });
    act(() => {
      document.dispatchEvent(new Event('visibilitychange'));
    });
    flushDebounce();
    expect(cb).not.toHaveBeenCalled();
  });

  it('coalesces a burst of resume events into one callback', () => {
    // Browsers commonly fire visibility + focus + online back-to-back on
    // wake. The 250ms debounce must collapse them so the caller doesn't
    // refetch 3× in a row.
    const cb = jest.fn();
    renderHook(() => useResume(cb));

    act(() => {
      window.dispatchEvent(new Event('focus'));
      window.dispatchEvent(new Event('online'));
    });
    fireVisibilityVisible();
    flushDebounce();

    expect(cb).toHaveBeenCalledTimes(1);
  });

  it('does not attach listeners when disabled', () => {
    const cb = jest.fn();
    renderHook(() => useResume(cb, false));

    act(() => {
      window.dispatchEvent(new Event('focus'));
    });
    flushDebounce();
    expect(cb).not.toHaveBeenCalled();
  });

  it('detaches listeners on unmount', () => {
    const cb = jest.fn();
    const { unmount } = renderHook(() => useResume(cb));
    unmount();

    act(() => {
      window.dispatchEvent(new Event('focus'));
    });
    flushDebounce();
    expect(cb).not.toHaveBeenCalled();
  });

  it('uses the latest callback closure', () => {
    // The hook stores the callback in a ref so the listener never holds a
    // stale reference. Important when the caller closes over component
    // state that changes between renders.
    const first = jest.fn();
    const second = jest.fn();
    const { rerender } = renderHook(({ cb }: { cb: () => void }) => useResume(cb), {
      initialProps: { cb: first },
    });

    rerender({ cb: second });

    act(() => {
      window.dispatchEvent(new Event('focus'));
    });
    flushDebounce();
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });
});
