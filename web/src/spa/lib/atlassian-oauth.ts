/**
 * Atlassian OAuth 2.0 + PKCE helpers.
 *
 * Entirely client-side — no backend is involved. The client_id is public
 * (PKCE requires no client_secret). All token material lives in localStorage
 * via useConnections; this module only handles the cryptographic ceremony and
 * network calls to Atlassian's authorization and token endpoints.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ATLASSIAN_AUTH_URL = 'https://auth.atlassian.com/authorize';
const ATLASSIAN_TOKEN_URL = 'https://auth.atlassian.com/oauth/token';
const ATLASSIAN_RESOURCES_URL =
  'https://api.atlassian.com/oauth/token/accessible-resources';

// Read-only scopes for Confluence + Jira + offline refresh
const SCOPES =
  'read:confluence-content.all read:confluence-space.summary read:jira-work read:jira-user offline_access';

const SESSION_VERIFIER_KEY = 'wikis.oauth.atlassian.code_verifier';
const SESSION_STATE_KEY = 'wikis.oauth.atlassian.state';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface AtlassianTokens {
  access_token: string;
  refresh_token: string;
  /** Epoch milliseconds — Date.now() + expires_in * 1000 at mint time. */
  expires_at: number;
  scope: string;
}

export interface AccessibleResource {
  id: string;   // cloud_id used in API calls
  name: string; // e.g. "your-org.atlassian.net"
  url: string;
  scopes: string[];
}

// Internal shape of Atlassian's token response
interface AtlassianTokenResponse {
  access_token: string;
  refresh_token: string;
  expires_in: number; // seconds
  scope: string;
  token_type: string;
}

// ---------------------------------------------------------------------------
// PKCE utilities (browser-native Web Crypto API)
// ---------------------------------------------------------------------------

/** Encode an ArrayBuffer as a URL-safe base64 string (no padding). */
function base64UrlEncode(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

/** Generate a cryptographically random base64url string of `byteLength` bytes. */
function randomBase64Url(byteLength: number): string {
  const bytes = new Uint8Array(byteLength);
  crypto.getRandomValues(bytes);
  return base64UrlEncode(bytes.buffer);
}

/** SHA-256 hash of a UTF-8 string, returned as a base64url string. */
async function sha256Base64Url(plain: string): Promise<string> {
  const encoded = new TextEncoder().encode(plain);
  const digest = await crypto.subtle.digest('SHA-256', encoded);
  return base64UrlEncode(digest);
}

// ---------------------------------------------------------------------------
// Client ID resolution
// ---------------------------------------------------------------------------

function getClientId(): string {
  const id = process.env.NEXT_PUBLIC_ATLASSIAN_CLIENT_ID;
  if (!id) {
    throw new Error(
      'NEXT_PUBLIC_ATLASSIAN_CLIENT_ID is not set. ' +
        'Add it to your .env file to enable Atlassian OAuth.',
    );
  }
  return id;
}

function getRedirectUri(): string {
  return `${window.location.origin}/oauth/atlassian/callback`;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Step 1 — Start the OAuth flow.
 *
 * Generates a code_verifier and state, stores them in sessionStorage, then
 * opens the Atlassian authorization URL in a popup window.
 */
export async function startAtlassianOAuth(): Promise<void> {
  const clientId = getClientId();
  const verifier = randomBase64Url(32);
  const state = randomBase64Url(16);
  const challenge = await sha256Base64Url(verifier);

  sessionStorage.setItem(SESSION_VERIFIER_KEY, verifier);
  sessionStorage.setItem(SESSION_STATE_KEY, state);

  const params = new URLSearchParams({
    audience: 'api.atlassian.com',
    client_id: clientId,
    scope: SCOPES,
    redirect_uri: getRedirectUri(),
    state,
    response_type: 'code',
    code_challenge: challenge,
    code_challenge_method: 'S256',
    prompt: 'consent',
  });

  const authUrl = `${ATLASSIAN_AUTH_URL}?${params.toString()}`;

  // Open as a popup so the wizard page can listen for the postMessage
  const popup = window.open(
    authUrl,
    'atlassian-oauth',
    'width=600,height=700,resizable=yes,scrollbars=yes',
  );

  if (!popup) {
    throw new Error(
      'Popup was blocked. Please allow popups for this site and try again.',
    );
  }
}

/**
 * Step 2 — Complete the OAuth flow inside the callback page.
 *
 * Reads the stored verifier from sessionStorage, exchanges the authorization
 * code with Atlassian's token endpoint, clears PKCE material, and returns the
 * token set.
 */
export async function completeAtlassianOAuth(
  code: string,
  state: string,
): Promise<AtlassianTokens> {
  const storedState = sessionStorage.getItem(SESSION_STATE_KEY);
  const verifier = sessionStorage.getItem(SESSION_VERIFIER_KEY);

  // CSRF / replay guard
  if (!storedState || storedState !== state) {
    throw new Error(
      'OAuth state mismatch — the request may have been tampered with.',
    );
  }
  if (!verifier) {
    throw new Error('PKCE code_verifier not found — session may have expired.');
  }

  // Clean up before the network call (no retry on failure intentionally)
  sessionStorage.removeItem(SESSION_STATE_KEY);
  sessionStorage.removeItem(SESSION_VERIFIER_KEY);

  const body = new URLSearchParams({
    grant_type: 'authorization_code',
    client_id: getClientId(),
    code,
    redirect_uri: getRedirectUri(),
    code_verifier: verifier,
  });

  const response = await fetch(ATLASSIAN_TOKEN_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(
      `Atlassian token exchange failed (${response.status}): ${text}`,
    );
  }

  const data: AtlassianTokenResponse = await response.json();
  return tokensFromResponse(data);
}

/**
 * Step 3 — Refresh an expired access token.
 *
 * Atlassian rotates refresh tokens on every use — always store the new one
 * returned in the response.
 */
export async function refreshAtlassianTokens(
  refreshToken: string,
): Promise<AtlassianTokens> {
  const body = new URLSearchParams({
    grant_type: 'refresh_token',
    client_id: getClientId(),
    refresh_token: refreshToken,
  });

  const response = await fetch(ATLASSIAN_TOKEN_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(
      `Atlassian token refresh failed (${response.status}): ${text}`,
    );
  }

  const data: AtlassianTokenResponse = await response.json();
  return tokensFromResponse(data);
}

/**
 * Fetch the list of Atlassian Cloud sites accessible under the given token.
 *
 * Used immediately after completing the OAuth flow to discover the cloud_id
 * for the user's site(s).
 */
export async function fetchAccessibleResources(
  accessToken: string,
): Promise<AccessibleResource[]> {
  const response = await fetch(ATLASSIAN_RESOURCES_URL, {
    headers: { Authorization: `Bearer ${accessToken}`, Accept: 'application/json' },
  });

  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(
      `Failed to fetch Atlassian accessible resources (${response.status}): ${text}`,
    );
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function tokensFromResponse(data: AtlassianTokenResponse): AtlassianTokens {
  return {
    access_token: data.access_token,
    refresh_token: data.refresh_token,
    expires_at: Date.now() + data.expires_in * 1000,
    scope: data.scope,
  };
}

// ---------------------------------------------------------------------------
// Exports for testing (not part of the public integration contract)
// ---------------------------------------------------------------------------
export { base64UrlEncode, sha256Base64Url, randomBase64Url };
