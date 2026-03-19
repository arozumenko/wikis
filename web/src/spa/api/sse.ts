import type { components } from './types.generated';
import { getAuthToken } from './client';

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
  | ({ type: 'tool_end' } & ToolEndEvent);

export type ResearchSSEEvent =
  | ({ type: 'research_start' } & { session_id: string; question: string; timestamp: string })
  | ({ type: 'thinking_step' } & ThinkingStepEvent)
  | ({ type: 'answer_chunk' } & { chunk: string; timestamp: string })
  | ({ type: 'research_complete' } & { session_id: string; report: string; timestamp: string })
  | ({ type: 'research_error' } & { session_id: string; error: string; timestamp: string })
  | ({ type: 'task_complete' } & { taskId?: string; status: string; wikiId?: string; wiki_id?: string })
  | ({ type: 'task_failed' } & { taskId?: string; status: string; error: string; recoverable?: boolean })
  | ({ type: 'todo_update' } & { todos: unknown[]; timestamp: string });

export type AskSSEEvent =
  | ({ type: 'ask_start' } & { session_id: string; question: string; timestamp: string })
  | ({ type: 'thinking_step' } & ThinkingStepEvent)
  | ({ type: 'answer_chunk' } & { chunk: string; timestamp: string })
  | ({ type: 'ask_complete' } & {
      answer: string;
      sources: Array<{
        file_path: string;
        snippet?: string;
        line_start?: number;
        line_end?: number;
      }>;
      steps?: number;
    })
  | ({ type: 'task_complete' } & { taskId?: string; status: string })
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
 * Uses fetch instead of native EventSource to support Authorization headers.
 * Returns an object with a close() method for compatibility with existing callers.
 */
export function subscribeSSE(
  path: string,
  onEvent: (event: SSEEventData) => void,
  onError?: (error: Event | Error) => void,
): { close: () => void } {
  const controller = new AbortController();

  (async () => {
    try {
      const token = await getAuthToken();
      const headers: Record<string, string> = {};
      if (token) headers['Authorization'] = `Bearer ${token}`;

      const resp = await fetch(`${API_BASE}${path}`, {
        headers,
        signal: controller.signal,
      });

      if (!resp.ok || !resp.body) {
        onError?.(new Error(`HTTP ${resp.status}`));
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
                onEvent({ type, ...payload } as SSEEventData);
              } catch {
                /* skip malformed */
              }
            }
            eventType = '';
            dataStr = '';
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        onError?.(e as Error);
      }
    }
  })();

  return { close: () => controller.abort() };
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
