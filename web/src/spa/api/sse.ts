import type { components } from './types.generated';
import { getAuthToken } from './client';

type SourceReference = components['schemas']['SourceReference'];

type ProgressEvent = components['schemas']['ProgressEvent'];
type PageCompleteEvent = components['schemas']['PageCompleteEvent'];
type WikiCompleteEvent = components['schemas']['WikiCompleteEvent'];
type ErrorEvent = components['schemas']['ErrorEvent'];

interface RetryEvent {
  event: string;
  message: string;
  attempt: number;
  max_attempts: number;
  wait_seconds: number;
}

interface FallbackEvent {
  event: string;
  message: string;
  from_model: string;
  to_model: string;
}

interface ToolStartEvent {
  tool: string;
  input: string;
  timestamp: string;
}

interface ToolEndEvent {
  tool: string;
  result_count?: number;
  output?: string;
  timestamp: string;
}

export interface ThinkingStepEvent {
  step: number;
  /** 'tool_call' or 'tool_result' — remapped from the backend 'type' field (legacy) */
  step_type?: 'tool_call' | 'tool_result';
  /** 'tool_call' or 'tool_result' — native camelCase field (new MCP format) */
  stepType?: 'tool_call' | 'tool_result';
  tool: string;
  tool_call_id?: string;
  toolCallId?: string;
  call_id?: string;
  callId?: string;
  input?: string;
  output?: string;
  output_preview?: string;
  outputPreview?: string;
  output_length?: number;
  outputLength?: number;
  timestamp: string;
}

/** A single todo item from the agent's TodoListMiddleware. */
export interface TodoItem {
  /** Backend sends 'content'; we also accept 'task' for flexibility */
  content?: string;
  task?: string;
  status: 'not_started' | 'in_progress' | 'completed';
}

/** Merged view of a single tool invocation (start + optional end). */
export interface ToolCallRecord {
  tool_name: string;
  tool_input: string;
  tool_output: string | null;
  timestamp: string;
  endTimestamp: string | null;
  done: boolean;
  /**
   * LangGraph / MCP tool call identifier. Carried through so the UI can match
   * `tool_result` events to the originating `tool_call` record by id rather
   * than by tool name + insertion order — this prevents off-by-one display
   * when the agent runs tool calls in parallel or returns them out of order.
   */
  tool_call_id?: string;
}

export type SSEEventData =
  | ({ type: 'progress' } & ProgressEvent & {
      // New MCP format fields
      progressToken?: string;
      total?: number;
      _phase?: string;
      _pageId?: string;
    })
  | ({ type: 'page_complete' } & PageCompleteEvent & {
      // New MCP format fields (camelCase)
      _pageId?: string;
      _pageTitle?: string;
    })
  | ({ type: 'task_complete' } & {
      taskId?: string;
      status: string;
      wikiId?: string;
      wiki_id?: string;
      pageCount?: number;
      page_count?: number;
      executionTime?: number;
      execution_time?: number;
    })
  | ({ type: 'wiki_complete' } & WikiCompleteEvent) // legacy
  | ({ type: 'task_failed' } & {
      taskId?: string;
      status: string;
      error: string;
      recoverable?: boolean;
      errorType?: string;
      error_type?: string;
      modelLimit?: number;
      model_limit?: number;
      suggestedActions?: string[];
      suggested_actions?: string[];
    })
  | ({ type: 'task_cancelled' } & { taskId?: string; status: string; statusMessage?: string })
  | ({ type: 'error' } & ErrorEvent) // legacy
  | ({ type: 'retry' } & RetryEvent)
  | ({ type: 'fallback' } & FallbackEvent)
  | ({ type: 'message' } & {
      level: string;
      data: string;
      tool?: string;
      attempt?: number;
      maxAttempts?: number;
      max_attempts?: number;
      waitSeconds?: number;
      wait_seconds?: number;
      fromModel?: string;
      from_model?: string;
      toModel?: string;
      to_model?: string;
    })
  | ({ type: 'tool_start' } & ToolStartEvent)
  | ({ type: 'tool_end' } & ToolEndEvent)
  // #116 PR 5: incremental refresh events. Per-page (one per regime)
  // plus a summary at the end of the run.
  | ({ type: 'page_unchanged' } & {
      progressToken?: string;
      _pageId?: string;
      _pageTitle?: string;
      timestamp?: string;
    })
  | ({ type: 'page_patched' } & {
      progressToken?: string;
      _pageId?: string;
      _pageTitle?: string;
      citationCount?: number;
      timestamp?: string;
    })
  | ({ type: 'page_edited' } & {
      progressToken?: string;
      _pageId?: string;
      _pageTitle?: string;
      diffRatio?: number;
      timestamp?: string;
    })
  | ({ type: 'page_regenerated' } & {
      progressToken?: string;
      _pageId?: string;
      _pageTitle?: string;
      demotedFromEdit?: boolean;
      timestamp?: string;
    })
  | ({ type: 'page_deleted' } & {
      progressToken?: string;
      _pageId?: string;
      _pageTitle?: string;
      timestamp?: string;
    })
  | ({ type: 'incremental_summary' } & {
      progressToken?: string;
      stats: IncrementalRegenStats;
      timestamp?: string;
    });

/**
 * Shape of the summary event payload — matches the Python
 * `IncrementalRegenStats.as_dict()` keys.
 */
export interface IncrementalRegenStats {
  total_pages: number;
  unchanged: number;
  trivial_patched: number;
  edit_applied: number;
  edit_demoted_to_structural: number;
  structural_regenerated: number;
  structural_failed: number;
  /** #141: pages whose entire cluster vanished. */
  deleted: number;
  /** #134: per-page reasons captured from the StructuralHandler. */
  structural_failure_reasons: string[];
  avg_diff_ratio: number;
}

export type ResearchSSEEvent =
  | ({ type: 'research_start' } & { session_id: string; question: string; timestamp: string })
  | ({ type: 'thinking_step' } & ThinkingStepEvent)
  | ({ type: 'answer_chunk' } & { chunk: string; timestamp: string })
  | ({ type: 'research_complete' } & { session_id: string; report: string; timestamp: string; code_map?: unknown })
  | ({ type: 'research_error' } & { session_id: string; error: string; timestamp: string })
  | ({ type: 'task_complete' } & { taskId?: string; status: string; wikiId?: string; wiki_id?: string })
  | ({ type: 'task_failed' } & { taskId?: string; status: string; error: string; recoverable?: boolean })
  | ({ type: 'todo_update' } & { todos: unknown[]; timestamp: string })
  | ({ type: 'code_map_ready' } & { code_map: unknown });

export type AskSSEEvent =
  | ({ type: 'ask_start' } & { session_id: string; question: string; timestamp: string })
  | ({ type: 'thinking_step' } & ThinkingStepEvent)
  | ({ type: 'answer_chunk' } & { chunk: string; timestamp: string })
  | ({ type: 'ask_complete' } & {
      answer: string;
      sources: SourceReference[];
      steps?: number;
    })
  | ({ type: 'task_complete' } & {
      taskId?: string;
      status: string;
      answer?: string;
      sources?: SourceReference[];
      steps?: number;
    })
  | ({ type: 'task_failed' } & { taskId?: string; status: string; error: string; recoverable?: boolean })
  | ({ type: 'ask_error' } & { error: string });

const API_BASE = ''; // Same origin — proxied through Next.js rewrites

/**
 * Detect and normalize both the legacy flat format and the new MCP JSON-RPC 2.0 format.
 *
 * New format: `{"jsonrpc": "2.0", "method": "notifications/progress", "params": {...}}`
 * Legacy format: flat data object
 *
 * Returns the event type (unchanged) and the normalized payload.
 */
function parseSSEData(
  eventType: string,
  raw: Record<string, unknown>,
): { type: string; payload: Record<string, unknown> } {
  if (raw.jsonrpc === '2.0' && raw.params) {
    const params = raw.params as Record<string, unknown>;
    return { type: eventType, payload: params };
  }
  return { type: eventType, payload: raw };
}

/**
 * Handle thinking_step backward compat: legacy backend sends `type: 'tool_call'/'tool_result'`
 * inside the data payload. Remap it to `step_type` to avoid collision with the SSE discriminated
 * union `type` field. New format sends `stepType` directly — no remapping needed.
 */
function normalizeThinkingStep(payload: Record<string, unknown>): Record<string, unknown> {
  // New format: stepType present — also set step_type for components that read snake_case
  if (payload.stepType) return { ...payload, step_type: payload.stepType };
  // Legacy format: `type` field holds the step kind — remap to step_type
  if (payload.type) {
    const { type: stepType, ...rest } = payload;
    return { ...rest, step_type: stepType };
  }
  return payload;
}

/**
 * Stream wiki generation SSE events via fetch + ReadableStream.
 *
 * Uses fetch (not native EventSource) so we can attach Authorization headers
 * and set ``Last-Event-ID`` on reconnect.
 *
 * #191: the stream survives transient drops — laptop sleep, brief offline
 * blips, idle TCP timeouts. The backend (``/api/v1/invocations/{id}/stream``)
 * honors ``Last-Event-ID`` for replay, so a reconnect resumes from the last
 * event we saw rather than re-replaying the full history. Backoff doubles
 * from 1s → 2s → 5s → 10s (capped) and resets to 1s on the first event of
 * a successful (re)connect. ``close()`` is sticky: once called, no further
 * reconnect attempts are made.
 */
export function subscribeSSE(
  path: string,
  onEvent: (event: SSEEventData) => void,
  onError?: (error: Event | Error) => void,
): { close: () => void } {
  const BACKOFF_MS = [1000, 2000, 5000, 10000];
  let closed = false;
  let lastEventId: string | null = null;
  let backoffIndex = 0;
  let controller: AbortController | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  // #191 (Copilot C6): if a clean close yields zero events on a connection
  // cycle AND we have previously received an event, treat the stream as
  // terminal and stop reconnecting. Without this, consumers that forget to
  // call close() on a terminal SSE event (e.g. IncrementalRefreshBanner
  // handles incremental_summary but doesn't close) would reconnect forever
  // — the backend keeps returning empty replies for a terminated invocation.
  let hasEverReceivedEvent = false;

  const scheduleReconnect = () => {
    if (closed) return;
    const delay = BACKOFF_MS[Math.min(backoffIndex, BACKOFF_MS.length - 1)];
    backoffIndex += 1;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      void connect();
    }, delay);
  };

  const connect = async (): Promise<void> => {
    if (closed) return;
    controller = new AbortController();
    try {
      const token = await getAuthToken();
      const headers: Record<string, string> = {};
      if (token) headers['Authorization'] = `Bearer ${token}`;
      if (lastEventId) headers['Last-Event-ID'] = lastEventId;

      const resp = await fetch(`${API_BASE}${path}`, {
        headers,
        signal: controller.signal,
      });

      if (!resp.ok || !resp.body) {
        // 401/403/404 = terminal (auth / not found). Surface to onError
        // so callers can render an error state, then stop.
        if (resp.status === 404 || resp.status === 401 || resp.status === 403) {
          closed = true;
          onError?.(new Error(`HTTP ${resp.status}`));
          return;
        }
        // #191 (Copilot C3): 5xx and other non-2xx are transient — the
        // server may be restarting. Reconnect silently; do NOT call
        // onError here. Consumers like IncrementalRefreshBanner treat any
        // onError as a permanent failure UI.
        scheduleReconnect();
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let eventType = '';
      let dataStr = '';
      let eventIdField = '';
      let firstEventSeenThisCycle = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        const lines = buf.split('\n');
        buf = lines.pop() ?? '';

        for (const rawLine of lines) {
          const line = rawLine.replace(/\r$/, '');
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            dataStr += (dataStr ? '\n' : '') + line.slice(6);
          } else if (line.startsWith('id: ')) {
            eventIdField = line.slice(4).trim();
          } else if (line === '') {
            if (eventType && dataStr) {
              try {
                const raw = JSON.parse(dataStr) as Record<string, unknown>;
                const { type, payload } = parseSSEData(eventType, raw);
                if (eventIdField) lastEventId = eventIdField;
                if (!firstEventSeenThisCycle) {
                  firstEventSeenThisCycle = true;
                  hasEverReceivedEvent = true;
                  backoffIndex = 0;
                }
                onEvent({ type, ...payload } as SSEEventData);
              } catch {
                /* skip malformed */
              }
            }
            eventType = '';
            dataStr = '';
            eventIdField = '';
          }
        }
      }
      if (closed) return;
      // Stream ended cleanly. If this cycle delivered zero events and we
      // had received events on a prior cycle, the producer is done — stop
      // reconnecting. Otherwise treat as a transient drop and reconnect.
      if (hasEverReceivedEvent && !firstEventSeenThisCycle) return;
      scheduleReconnect();
    } catch (e) {
      if ((e as Error).name === 'AbortError') return;
      // #191 (Copilot C3): silent reconnect on transient network errors.
      // onError is reserved for terminal failures (HTTP 4xx, surfaced
      // above) so reconnecting consumers don't see a false "failed" UI.
      if (!closed) scheduleReconnect();
    }
  };

  void connect();

  return {
    close: () => {
      closed = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      controller?.abort();
    },
  };
}

/**
 * Stream research events over a POST request (fetch + ReadableStream).
 * Returns a cancel function that aborts the request.
 */
export function subscribeResearchSSE(
  path: string,
  body: unknown,
  onEvent: (event: ResearchSSEEvent) => void,
  onDone: () => void,
  onError: (e: unknown) => void,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const token = await getAuthToken();
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      if (token) headers['Authorization'] = `Bearer ${token}`;

      const resp = await fetch(`${API_BASE}${path}?accept=text/event-stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!resp.ok || !resp.body) {
        onError(new Error(`HTTP ${resp.status}`));
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let eventType = '';
      let dataStr = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        const lines = buf.split('\n');
        buf = lines.pop() ?? '';

        for (const rawLine of lines) {
          const line = rawLine.replace(/\r$/, '');
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            dataStr += (dataStr ? '\n' : '') + line.slice(6);
          } else if (line === '') {
            if (eventType && dataStr) {
              try {
                const raw = JSON.parse(dataStr) as Record<string, unknown>;
                const { type, payload: basePayload } = parseSSEData(eventType, raw);
                const payload =
                  type === 'thinking_step' ? normalizeThinkingStep(basePayload) : basePayload;
                onEvent({ type, ...payload } as ResearchSSEEvent);
              } catch {
                /* skip malformed */
              }
            }
            eventType = '';
            dataStr = '';
          }
        }
      }

      onDone();
    } catch (e) {
      if ((e as Error).name !== 'AbortError') onError(e);
    }
  })();

  return () => controller.abort();
}

/**
 * Stream ask (fast Q&A) events over a POST request.
 * Receives tool_start/tool_end while the agent runs, then ask_complete with the final answer.
 * Returns a cancel function that aborts the request.
 */
export function subscribeAskSSE(
  path: string,
  body: unknown,
  onEvent: (event: AskSSEEvent) => void,
  onDone: () => void,
  onError: (e: unknown) => void,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const token = await getAuthToken();
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      if (token) headers['Authorization'] = `Bearer ${token}`;

      const resp = await fetch(`${API_BASE}${path}?accept=text/event-stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!resp.ok || !resp.body) {
        onError(new Error(`HTTP ${resp.status}`));
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let eventType = '';
      let dataStr = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        const lines = buf.split('\n');
        buf = lines.pop() ?? '';

        for (const rawLine of lines) {
          const line = rawLine.replace(/\r$/, '');
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            dataStr += (dataStr ? '\n' : '') + line.slice(6);
          } else if (line === '') {
            if (eventType && dataStr) {
              try {
                const raw = JSON.parse(dataStr) as Record<string, unknown>;
                const { type, payload: basePayload } = parseSSEData(eventType, raw);
                const payload =
                  type === 'thinking_step' ? normalizeThinkingStep(basePayload) : basePayload;
                onEvent({ type, ...payload } as AskSSEEvent);
              } catch {
                /* skip malformed */
              }
            }
            eventType = '';
            dataStr = '';
          }
        }
      }

      onDone();
    } catch (e) {
      if ((e as Error).name !== 'AbortError') onError(e);
    }
  })();

  return () => controller.abort();
}

// ---------------------------------------------------------------------
// PR-16 — Project recompute SSE (cross-repo pipeline progress)
// ---------------------------------------------------------------------

export interface RecomputeProgressEvent {
  type:
    | 'project_relatedness_progress'
    | 'cross_repo_linker_progress'
    | 'project_clustering_progress'
    | 'recompute_complete'
    | 'recompute_error'
    | string;
  progressToken?: string;
  progress?: number;
  total?: number;
  message?: string;
  _phase?: string;
  _projectId?: string;
  _pair?: [string, string];
  _communityCount?: number;
  // recompute_complete payload
  status?: string;
  wiki_count?: number;
  pair_count?: number;
  edge_count?: number;
  community_count?: number;
  recomputed_at?: string;
  // recompute_error payload
  error?: string;
}

/**
 * Subscribe to a project recompute SSE stream
 * (POST /api/v1/projects/{projectId}/recompute).
 *
 * Returns a cancel function that aborts the request.
 */
export function subscribeRecomputeSSE(
  projectId: string,
  onEvent: (event: RecomputeProgressEvent) => void,
  onDone: () => void,
  onError: (e: unknown) => void,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const token = await getAuthToken();
      const headers: Record<string, string> = {
        Accept: 'text/event-stream',
      };
      if (token) headers['Authorization'] = `Bearer ${token}`;

      const resp = await fetch(
        `${API_BASE}/api/v1/projects/${encodeURIComponent(projectId)}/recompute`,
        { method: 'POST', headers, signal: controller.signal },
      );

      if (!resp.ok || !resp.body) {
        onError(new Error(`HTTP ${resp.status}`));
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let eventType = '';
      let dataStr = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        const lines = buf.split('\n');
        buf = lines.pop() ?? '';

        for (const rawLine of lines) {
          const line = rawLine.replace(/\r$/, '');
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            dataStr += (dataStr ? '\n' : '') + line.slice(6);
          } else if (line === '') {
            if (eventType && dataStr) {
              try {
                const raw = JSON.parse(dataStr) as Record<string, unknown>;
                const { type, payload } = parseSSEData(eventType, raw);
                onEvent({ type, ...payload } as RecomputeProgressEvent);
              } catch {
                /* skip malformed */
              }
            }
            eventType = '';
            dataStr = '';
          }
        }
      }

      onDone();
    } catch (e) {
      if ((e as Error).name !== 'AbortError') onError(e);
    }
  })();

  return () => controller.abort();
}
