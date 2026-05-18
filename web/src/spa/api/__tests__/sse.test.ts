/**
 * #191 — subscribeSSE must survive transient connection drops. The
 * frontend treats one network blip (laptop sleep, idle TCP timeout, brief
 * offline) as a transient — it reconnects with backoff and resumes from
 * the last seen event id. User-initiated close() must stop reconnects.
 */

// Polyfills for the node test env: jsdom isn't a great fit here because
// jest's jsdom env doesn't ship ReadableStream / TextEncoder consistently.
// The node env has both natively (Node ≥ 18) — fine for this isolated
// unit test that only exercises fetch + the read loop.

jest.mock('../client', () => ({
  getAuthToken: jest.fn().mockResolvedValue('mock-token'),
}));

import { subscribeSSE } from '../sse';

// ---------------------------------------------------------------------------
// Controllable mock body — push events / end / error from the test thread
// ---------------------------------------------------------------------------

type Pushable = {
  push: (chunk: string) => void;
  end: () => void;
};

function makePushableBody(): { body: ReadableStream<Uint8Array>; ctl: Pushable } {
  const encoder = new TextEncoder();
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c;
    },
  });
  return {
    body,
    ctl: {
      push: (chunk) => controller.enqueue(encoder.encode(chunk)),
      end: () => controller.close(),
    },
  };
}

function eventFrame(type: string, data: unknown, id?: string | number): string {
  const lines = [`event: ${type}`];
  if (id !== undefined) lines.push(`id: ${id}`);
  lines.push(`data: ${JSON.stringify(data)}`);
  return lines.join('\n') + '\n\n';
}

// Real timers, with helper to wait for a backoff-window plus microtask drain.
async function drainMicrotasks(times = 5) {
  for (let i = 0; i < times; i++) {
    await Promise.resolve();
  }
}

async function waitMs(ms: number) {
  // Real-time sleep with microtask drains on either side so any pending
  // .then() chains land before the next assertion.
  await drainMicrotasks();
  await new Promise((r) => setTimeout(r, ms));
  await drainMicrotasks();
}

describe('subscribeSSE', () => {
  let fetchMock: jest.Mock;

  beforeEach(() => {
    fetchMock = jest.fn();
    (global as unknown as { fetch: jest.Mock }).fetch = fetchMock;
  });

  afterEach(() => {
    delete (global as unknown as { fetch?: unknown }).fetch;
  });

  it('parses event-stream into typed events', async () => {
    const { body, ctl } = makePushableBody();
    fetchMock.mockResolvedValueOnce({ ok: true, status: 200, body });

    const events: unknown[] = [];
    const sub = subscribeSSE('/path', (e) => events.push(e));

    await drainMicrotasks();
    ctl.push(eventFrame('progress', { event: 'progress', message: 'hello' }, 1));
    await drainMicrotasks(10);

    expect(events).toEqual([
      expect.objectContaining({ type: 'progress', message: 'hello' }),
    ]);

    sub.close();
    ctl.end();
  });

  it('reconnects with Last-Event-ID on transient stream end', async () => {
    const first = makePushableBody();
    const second = makePushableBody();
    fetchMock
      .mockResolvedValueOnce({ ok: true, status: 200, body: first.body })
      .mockResolvedValueOnce({ ok: true, status: 200, body: second.body });

    const events: unknown[] = [];
    const sub = subscribeSSE('/path', (e) => events.push(e));

    await drainMicrotasks();
    first.ctl.push(eventFrame('progress', { event: 'progress', message: 'm1' }, 5));
    await drainMicrotasks(10);
    first.ctl.end();

    // First reconnect attempt: 1s backoff.
    await waitMs(1500);

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const secondCallHeaders = (fetchMock.mock.calls[1][1] as { headers: Record<string, string> })
      .headers;
    expect(secondCallHeaders['Last-Event-ID']).toBe('5');

    second.ctl.push(eventFrame('progress', { event: 'progress', message: 'm2' }, 6));
    await drainMicrotasks(10);

    expect(events).toEqual([
      expect.objectContaining({ message: 'm1' }),
      expect.objectContaining({ message: 'm2' }),
    ]);

    sub.close();
    second.ctl.end();
  });

  it('close() suppresses further reconnect attempts', async () => {
    const first = makePushableBody();
    fetchMock.mockResolvedValueOnce({ ok: true, status: 200, body: first.body });

    const sub = subscribeSSE('/path', () => {});
    await drainMicrotasks();

    sub.close();
    first.ctl.end();
    await waitMs(1500); // well past the first backoff slot

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('does not reconnect on terminal 4xx (404)', async () => {
    fetchMock.mockResolvedValueOnce({ ok: false, status: 404, body: null });
    const onError = jest.fn();

    subscribeSSE('/path', () => {}, onError);
    await waitMs(1500); // past first backoff slot

    expect(onError).toHaveBeenCalledWith(expect.any(Error));
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('reconnect backoff resets after a successful event', async () => {
    const a = makePushableBody();
    const b = makePushableBody();
    const c = makePushableBody();
    fetchMock
      .mockResolvedValueOnce({ ok: true, status: 200, body: a.body })
      .mockResolvedValueOnce({ ok: true, status: 200, body: b.body })
      .mockResolvedValueOnce({ ok: true, status: 200, body: c.body });

    const sub = subscribeSSE('/path', () => {});
    await drainMicrotasks();

    // First connection drops with no events — backoff index advances.
    a.ctl.end();
    await waitMs(1500);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    // Second connection delivers one event, then drops. Backoff resets.
    b.ctl.push(eventFrame('progress', { event: 'progress', message: 'ok' }, 1));
    await drainMicrotasks(10);
    b.ctl.end();

    // The next reconnect must fire after 1s, not 2s. Wait just 1.1s.
    await waitMs(1500);
    expect(fetchMock).toHaveBeenCalledTimes(3);

    sub.close();
    c.ctl.end();
  });

  it('does not reconnect on terminal 401', async () => {
    fetchMock.mockResolvedValueOnce({ ok: false, status: 401, body: null });
    const onError = jest.fn();

    subscribeSSE('/path', () => {}, onError);
    await waitMs(1500);

    expect(onError).toHaveBeenCalledWith(expect.any(Error));
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  // Copilot C6 — stop reconnecting after a clean close that delivered zero
  // events on a cycle where we had previously received events. Without this
  // guard, consumers that forget to call close() on terminal SSE events
  // would loop forever against a terminated invocation.
  it('stops reconnecting when a clean close yields zero events after prior events', async () => {
    const a = makePushableBody();
    const b = makePushableBody();
    fetchMock
      .mockResolvedValueOnce({ ok: true, status: 200, body: a.body })
      .mockResolvedValueOnce({ ok: true, status: 200, body: b.body });

    const events: unknown[] = [];
    const sub = subscribeSSE('/path', (e) => events.push(e));
    await drainMicrotasks();

    // Cycle 1 delivers an event then ends.
    a.ctl.push(eventFrame('progress', { event: 'progress', message: 'one' }, 1));
    await drainMicrotasks(10);
    a.ctl.end();

    await waitMs(1500);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    // Cycle 2 ends immediately with no events — terminal. No 3rd fetch.
    b.ctl.end();
    await waitMs(2500);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    sub.close();
  });

  // Copilot C3 — transient (5xx, network throw) must not surface to onError.
  // Existing consumers (e.g. IncrementalRefreshBanner) treat any onError as
  // a permanent failure UI; a sleep/offline blip would falsely render that.
  it('does not call onError on transient 5xx; reconnects silently', async () => {
    const ok = makePushableBody();
    fetchMock
      .mockResolvedValueOnce({ ok: false, status: 503, body: null })
      .mockResolvedValueOnce({ ok: true, status: 200, body: ok.body });

    const onError = jest.fn();
    const sub = subscribeSSE('/path', () => {}, onError);

    await waitMs(1500);

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(onError).not.toHaveBeenCalled();

    sub.close();
    ok.ctl.end();
  });
});
