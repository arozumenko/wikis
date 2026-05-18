/**
 * Tests for the PKCE helper functions in atlassian-oauth.ts.
 *
 * We only test the pure crypto utilities — the OAuth network calls are
 * exercised in integration/E2E tests.
 */

import { base64UrlEncode, sha256Base64Url, randomBase64Url } from '../atlassian-oauth';

// ---------------------------------------------------------------------------
// base64UrlEncode
// ---------------------------------------------------------------------------

describe('base64UrlEncode', () => {
  it('produces a string without +, /, or = characters', () => {
    // All 256 possible bytes
    const all256 = new Uint8Array(256);
    for (let i = 0; i < 256; i++) all256[i] = i;
    const encoded = base64UrlEncode(all256.buffer);
    expect(encoded).not.toMatch(/[+/=]/);
  });

  it('round-trips known bytes', () => {
    // "hello" → base64 is "aGVsbG8=" → base64url is "aGVsbG8"
    const bytes = new TextEncoder().encode('hello');
    expect(base64UrlEncode(bytes.buffer)).toBe('aGVsbG8');
  });

  it('returns a non-empty string for non-empty input', () => {
    const bytes = new Uint8Array([1, 2, 3]);
    expect(base64UrlEncode(bytes.buffer).length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// randomBase64Url
// ---------------------------------------------------------------------------

describe('randomBase64Url', () => {
  it('returns a string for any positive byte length', () => {
    const result = randomBase64Url(32);
    expect(typeof result).toBe('string');
    expect(result.length).toBeGreaterThan(0);
  });

  it('does not contain standard base64 padding or special chars', () => {
    const result = randomBase64Url(32);
    expect(result).not.toMatch(/[+/=]/);
  });

  it('produces different values on consecutive calls (statistical check)', () => {
    const a = randomBase64Url(16);
    const b = randomBase64Url(16);
    // Probability of collision is astronomically small (2^-128)
    expect(a).not.toBe(b);
  });
});

// ---------------------------------------------------------------------------
// sha256Base64Url
// ---------------------------------------------------------------------------

describe('sha256Base64Url', () => {
  it('returns a non-empty base64url string', async () => {
    const result = await sha256Base64Url('hello world');
    expect(typeof result).toBe('string');
    expect(result.length).toBeGreaterThan(0);
    expect(result).not.toMatch(/[+/=]/);
  });

  it('is deterministic for the same input', async () => {
    const a = await sha256Base64Url('test-verifier');
    const b = await sha256Base64Url('test-verifier');
    expect(a).toBe(b);
  });

  it('produces different hashes for different inputs', async () => {
    const a = await sha256Base64Url('verifier-a');
    const b = await sha256Base64Url('verifier-b');
    expect(a).not.toBe(b);
  });

  it('matches a known SHA-256 base64url value', async () => {
    // SHA-256("abc") = ba7816bf...  in hex
    // base64url of that digest = "ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0"
    const result = await sha256Base64Url('abc');
    expect(result).toBe('ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0');
  });
});
