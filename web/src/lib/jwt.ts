import { SignJWT, importPKCS8 } from 'jose';

interface JWTPayload {
  sub: string;
  email: string;
  name: string;
  provider: string;
  [key: string]: unknown;
}

/**
 * Issue an RS256-signed JWT for cross-service auth with the FastAPI backend.
 * The backend verifies with the corresponding public key.
 */
export async function issueJWT(payload: JWTPayload): Promise<string> {
  const privateKeyPem = process.env.JWT_PRIVATE_KEY;
  if (!privateKeyPem) {
    throw new Error('JWT_PRIVATE_KEY environment variable is not set');
  }

  const key = await importPKCS8(privateKeyPem, 'RS256');

  return new SignJWT(payload)
    .setProtectedHeader({ alg: 'RS256' })
    .setIssuedAt()
    .setIssuer('wikis-auth')
    .setExpirationTime('24h')
    .sign(key);
}
