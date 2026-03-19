import { proxyGetStream } from '../../../_lib/sse-proxy';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// Invocation IDs are hex SHA-256 fragments — reject anything else to prevent
// path-traversal SSRF (e.g. "../../admin" reaching arbitrary backend endpoints).
const SAFE_ID = /^[a-zA-Z0-9_-]{1,64}$/;

export async function GET(
  request: Request,
  { params }: { params: Promise<{ invocationId: string }> },
) {
  const { invocationId } = await params;
  if (!SAFE_ID.test(invocationId)) {
    return new Response('Invalid invocation ID', { status: 400 });
  }
  return proxyGetStream(request, `/api/v1/invocations/${invocationId}/stream`);
}
