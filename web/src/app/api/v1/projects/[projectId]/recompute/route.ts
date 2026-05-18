import { proxyPost } from '../../../_lib/sse-proxy';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// Project IDs are short identifiers — reject anything else to prevent
// path-traversal SSRF reaching arbitrary backend endpoints.
const SAFE_ID = /^[a-zA-Z0-9_-]{1,128}$/;

export async function POST(
  request: Request,
  { params }: { params: Promise<{ projectId: string }> },
) {
  const { projectId } = await params;
  if (!SAFE_ID.test(projectId)) {
    return new Response('Invalid project ID', { status: 400 });
  }
  return proxyPost(request, `/api/v1/projects/${projectId}/recompute`);
}
