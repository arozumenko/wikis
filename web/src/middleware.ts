import { NextRequest, NextResponse } from 'next/server';
import { getSessionCookie } from 'better-auth/cookies';

const BACKEND_URL = process.env.BACKEND_URL || 'http://backend:8000';

/**
 * SSE-streaming paths handled by App Router route handlers (src/app/api/v1/).
 * These must NOT be proxied via rewrite — the route handlers add no-buffer
 * headers that are critical for SSE to work through Next.js.
 */
const SSE_PATHS = [
  /^\/api\/v1\/invocations\/[^/]+\/stream$/,
  /^\/api\/v1\/ask$/,
  /^\/api\/v1\/research$/,
];

function isSSEPath(pathname: string): boolean {
  return SSE_PATHS.some((re) => re.test(pathname));
}

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // --- API proxy (runs at middleware step 3, before rewrites and routes) ---
  // Proxy non-SSE /api/v1/* requests to the backend. SSE paths are left
  // for App Router route handlers which add no-buffer headers.
  if (pathname.startsWith('/api/v1/') && !isSSEPath(pathname)) {
    const url = new URL(pathname + request.nextUrl.search, BACKEND_URL);
    return NextResponse.rewrite(url);
  }
  if (pathname.startsWith('/mcp')) {
    const url = new URL(pathname + request.nextUrl.search, BACKEND_URL);
    return NextResponse.rewrite(url);
  }
  if (pathname === '/health') {
    return NextResponse.rewrite(new URL('/health', BACKEND_URL));
  }

  // Any remaining /api/* paths (SSE route handlers, unknown endpoints) pass through.
  // This prevents the auth guard below from redirecting API requests to /login.
  // SSE route handlers handle their own auth via Authorization header forwarding.
  if (pathname.startsWith('/api/')) {
    return NextResponse.next();
  }

  // --- Auth guard (page routes only) ---
  const sessionCookie = getSessionCookie(request);

  // User on /login with a session cookie
  if (sessionCookie && pathname === '/login') {
    const callbackUrl = request.nextUrl.searchParams.get('callbackUrl');

    // If there's a callbackUrl, the user was bounced here by a 401 — the session
    // cookie is likely stale. Let them through to the login page so they can
    // re-authenticate. This breaks the redirect loop without needing DB validation.
    if (callbackUrl) {
      return NextResponse.next();
    }

    // No callbackUrl — user navigated to /login directly while authenticated.
    // Redirect them to the dashboard.
    return NextResponse.redirect(new URL('/', request.url));
  }

  // Unauthenticated user on protected page → redirect to /login
  if (!sessionCookie && pathname !== '/login' && pathname !== '/logout') {
    const loginUrl = new URL('/login', request.url);
    if (pathname !== '/') {
      loginUrl.searchParams.set('callbackUrl', pathname + request.nextUrl.search);
    }
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  // Match API paths (for proxying) and page paths (for auth guard).
  // Exclude _next/static, _next/image, favicon.
  // Note: /api/auth/* is excluded so Better-Auth routes aren't intercepted.
  // All other /api/* paths are handled by the proxy/passthrough logic above.
  matcher: ['/((?!_next/static|_next/image|favicon.ico|api/auth).*)'],
};
