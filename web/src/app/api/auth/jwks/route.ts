import { exportJWK, importSPKI } from 'jose';
import { NextResponse } from 'next/server';

/**
 * GET /api/auth/jwks
 *
 * Exposes the RS256 public key as a JWKS so the FastAPI backend
 * can verify JWTs without sharing the private key.
 */
export async function GET() {
  const publicKeyPem = process.env.JWT_PUBLIC_KEY;
  if (!publicKeyPem) {
    return NextResponse.json({ error: 'JWT_PUBLIC_KEY not configured' }, { status: 500 });
  }

  try {
    const publicKey = await importSPKI(publicKeyPem, 'RS256');
    const jwk = await exportJWK(publicKey);
    jwk.kid = 'wikis-auth-1';
    jwk.alg = 'RS256';
    jwk.use = 'sig';

    return NextResponse.json(
      { keys: [jwk] },
      {
        headers: {
          'Cache-Control': 'public, max-age=3600',
        },
      },
    );
  } catch {
    return NextResponse.json({ error: 'Failed to export JWKS' }, { status: 500 });
  }
}
