import { auth } from '@/lib/auth';
import { NextRequest, NextResponse } from 'next/server';

/**
 * POST /api/auth/api-key/verify
 *
 * Wraps Better Auth's server-side verifyApiKey (which has no HTTP route)
 * so the FastAPI backend can validate API keys via HTTP.
 *
 * Expected request:  { "key": "wikis_..." }
 * Success response:  { "valid": true, "userId": "..." }
 * Failure response:  { "valid": false, "error": { "message": "...", "code": "..." } }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const result = await auth.api.verifyApiKey({ body: { key: body.key } });

    if (!result.valid) {
      return NextResponse.json(result);
    }

    // Flatten userId to the top level — the backend expects { valid, userId }.
    // Better Auth nests it under key.referenceId (mapped to userId in our schema).
    const userId = result.key?.referenceId ?? '';
    return NextResponse.json({
      valid: true,
      userId,
    });
  } catch {
    return NextResponse.json({ valid: false, error: { message: 'Verification failed' } });
  }
}
