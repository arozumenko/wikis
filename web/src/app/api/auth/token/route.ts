import { NextResponse } from 'next/server';
import { headers } from 'next/headers';
import { auth } from '@/lib/auth';
import { issueJWT } from '@/lib/jwt';

/**
 * GET /api/auth/token
 *
 * Issues an RS256-signed JWT for the authenticated user.
 * The frontend sends this token to the FastAPI backend.
 */
export async function GET() {
  const session = await auth.api.getSession({
    headers: await headers(),
  });

  if (!session?.user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    // Detect provider from session — Better Auth stores it in session.session
    const provider = (session.session as Record<string, unknown>)?.providerId as string | undefined;

    const token = await issueJWT({
      sub: session.user.id,
      email: session.user.email ?? '',
      name: session.user.name ?? '',
      provider: provider ?? 'credentials',
    });

    return NextResponse.json({ token });
  } catch {
    return NextResponse.json({ error: 'Failed to issue token' }, { status: 500 });
  }
}
