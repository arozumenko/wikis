import { NextResponse } from 'next/server';
import { auth } from '@/lib/auth';

export async function POST(request: Request) {
  let body: { username?: string; password?: string; email?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
  }

  const { username, password, email } = body;

  if (!username || !password) {
    return NextResponse.json({ error: 'Username and password are required' }, { status: 422 });
  }

  // Validate username: 3-30 chars, alphanumeric + underscore
  if (!/^[a-zA-Z0-9_]{3,30}$/.test(username)) {
    return NextResponse.json(
      { error: 'Username must be 3-30 characters (letters, numbers, underscore)' },
      { status: 422 },
    );
  }

  // Validate password: min 8 chars, at least one letter + one number
  if (!/^(?=.*[a-zA-Z])(?=.*\d).{8,}$/.test(password)) {
    return NextResponse.json(
      { error: 'Password must be at least 8 characters with at least one letter and one number' },
      { status: 422 },
    );
  }

  try {
    // Use Better Auth's sign-up API
    const result = await auth.api.signUpEmail({
      body: {
        name: username,
        email: email || `${username}@localhost`,
        password,
      },
    });

    return NextResponse.json({ id: result.user.id, username: result.user.name }, { status: 201 });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Registration failed';
    if (message.includes('already exists') || message.includes('unique')) {
      return NextResponse.json({ error: 'Username already taken' }, { status: 409 });
    }
    return NextResponse.json({ error: message }, { status: 422 });
  }
}
