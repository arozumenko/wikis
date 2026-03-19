import { proxyPost } from '../_lib/sse-proxy';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  return proxyPost(request, '/api/v1/research');
}
