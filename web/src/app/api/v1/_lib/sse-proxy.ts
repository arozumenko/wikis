/**
 * Shared SSE proxy utility — pipes a streaming response from the FastAPI
 * backend through a Next.js route handler with proper no-buffer headers.
 */

const BACKEND_URL = process.env.BACKEND_URL || 'http://backend:8000';

const SSE_HEADERS: Record<string, string> = {
  'Content-Type': 'text/event-stream; charset=utf-8',
  'Cache-Control': 'no-cache, no-transform',
  'Connection': 'keep-alive',
  'Content-Encoding': 'none',
  'X-Accel-Buffering': 'no',
};

/** Build full backend URL for a given path + query string. */
export function backendUrl(path: string, search: string): string {
  return `${BACKEND_URL}${path}${search ? `?${search}` : ''}`;
}

/** Forward relevant headers from the incoming request. */
function forwardHeaders(req: Request): Record<string, string> {
  const headers: Record<string, string> = {};
  const auth = req.headers.get('authorization');
  if (auth) headers['Authorization'] = auth;
  const ct = req.headers.get('content-type');
  if (ct) headers['Content-Type'] = ct;
  return headers;
}

/**
 * Proxy a POST request to the backend, returning either an SSE stream
 * or a plain JSON response depending on what the backend sends back.
 */
export async function proxyPost(req: Request, path: string): Promise<Response> {
  const { search } = new URL(req.url);
  const url = backendUrl(path, search.replace(/^\?/, ''));

  const upstream = await fetch(url, {
    method: 'POST',
    headers: forwardHeaders(req),
    body: req.body,
    // @ts-expect-error -- duplex is required for streaming request body in Node 20
    duplex: 'half',
  });

  if (!upstream.ok) {
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { 'Content-Type': upstream.headers.get('content-type') || 'application/json' },
    });
  }

  const contentType = upstream.headers.get('content-type') || '';
  if (contentType.includes('text/event-stream')) {
    return new Response(upstream.body, { status: 200, headers: SSE_HEADERS });
  }

  // Non-streaming (JSON) — pass through as-is
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { 'Content-Type': contentType },
  });
}

/**
 * Proxy a GET request to the backend, returning an SSE stream.
 */
export async function proxyGetStream(req: Request, path: string): Promise<Response> {
  const url = backendUrl(path, '');

  const upstream = await fetch(url, {
    method: 'GET',
    headers: forwardHeaders(req),
  });

  if (!upstream.ok) {
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { 'Content-Type': upstream.headers.get('content-type') || 'application/json' },
    });
  }

  return new Response(upstream.body, { status: 200, headers: SSE_HEADERS });
}
