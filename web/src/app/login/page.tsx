'use client';

import { useSearchParams } from 'next/navigation';
import { FormEvent, Suspense, useCallback, useState } from 'react';
import {
  Box,
  Button,
  CircularProgress,
  CssBaseline,
  Divider,
  TextField,
  ThemeProvider,
  Typography,
} from '@mui/material';
import GitHubIcon from '@mui/icons-material/GitHub';
import GoogleIcon from '@mui/icons-material/Google';
import { createAppTheme } from '../../spa/theme';
import { useThemeMode } from '../../spa/hooks/useThemeMode';

const showGithub = process.env.NEXT_PUBLIC_SHOW_GITHUB !== 'false';
const showGoogle = process.env.NEXT_PUBLIC_SHOW_GOOGLE !== 'false';

function LoginForm() {
  const searchParams = useSearchParams();
  const error = searchParams.get('error');
  // Sanitize callbackUrl: only allow single-slash relative paths, never full URLs,
  // protocol-relative URLs (//evil.com), or /login (redirect loop)
  const rawCallback = searchParams.get('callbackUrl') ?? '/';
  const callbackUrl =
    rawCallback.startsWith('/') && !rawCallback.startsWith('//') && !rawCallback.startsWith('/login')
      ? rawCallback
      : '/';

  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [formError, setFormError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleCredentialsLogin = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      setFormError(null);
      setLoading(true);

      try {
        const resp = await fetch('/api/auth/sign-in/email', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password }),
          credentials: 'include',
        });

        if (resp.ok) {
          // Use assign() with absolute URL — most reliable cross-browser.
          // Safari sometimes ignores relative href assignments after fetch.
          window.location.assign(new URL(callbackUrl, window.location.origin).href);
          return;
        } else {
          setFormError('Invalid email or password');
        }
      } catch {
        setFormError('Sign in failed. Please try again.');
      }
      setLoading(false);
    },
    [email, password, callbackUrl],
  );

  const handleRegister = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      setFormError(null);
      setLoading(true);

      try {
        const resp = await fetch('/api/auth/sign-up/email', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name || email.split('@')[0], email, password }),
          credentials: 'include',
        });

        if (resp.ok) {
          window.location.assign(new URL(callbackUrl, window.location.origin).href);
          return;
        } else {
          const data = await resp.json().catch(() => ({}));
          setFormError(data.message ?? 'Registration failed');
        }
      } catch {
        setFormError('Registration failed. Please try again.');
      }
      setLoading(false);
    },
    [name, email, password, callbackUrl],
  );

  const handleOAuthSignIn = useCallback(
    (provider: string) => {
      window.location.href = `/api/auth/sign-in/social?provider=${provider}&callbackURL=${encodeURIComponent(callbackUrl)}`;
    },
    [callbackUrl],
  );

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        bgcolor: 'background.default',
      }}
    >
      <Box
        sx={{
          width: '100%',
          maxWidth: 400,
          px: 4,
          py: 5,
          mx: 'auto',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 3,
        }}
      >
        <Typography
          variant="h4"
          component="h1"
          sx={{ fontFamily: '"Playfair Display", Georgia, serif', fontWeight: 700, mb: 1, color: 'text.primary' }}
        >
          {mode === 'login' ? 'Sign in to Wikis' : 'Create Account'}
        </Typography>

        {(error || formError) && (
          <Typography color="error" variant="body2" sx={{ textAlign: 'center', maxWidth: 320 }}>
            {formError ?? 'An error occurred during sign in.'}
          </Typography>
        )}

        <Box
          component="form"
          onSubmit={mode === 'login' ? handleCredentialsLogin : handleRegister}
          sx={{ display: 'flex', flexDirection: 'column', gap: 2, width: '100%' }}
        >
          {mode === 'register' && (
            <TextField
              label="Display name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={loading}
              fullWidth
              size="medium"
              autoComplete="name"
            />
          )}
          <TextField
            label="Email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            disabled={loading}
            fullWidth
            size="medium"
            autoComplete="email"
          />
          <TextField
            label="Password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            disabled={loading}
            fullWidth
            size="medium"
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
          />
          <Button
            type="submit"
            variant="contained"
            size="large"
            disabled={loading}
            fullWidth
            sx={{ mt: 1, py: 1.3 }}
          >
            {loading ? (
              <CircularProgress size={22} color="inherit" />
            ) : mode === 'login' ? (
              'Sign In'
            ) : (
              'Create Account'
            )}
          </Button>
        </Box>

        <Button
          variant="text"
          size="small"
          onClick={() => {
            setMode(mode === 'login' ? 'register' : 'login');
            setFormError(null);
          }}
          sx={{ textTransform: 'none' }}
        >
          {mode === 'login' ? 'Create an account' : 'Already have an account? Sign in'}
        </Button>

        {(showGithub || showGoogle) && (
          <>
            <Divider sx={{ width: '100%' }}>
              <Typography variant="body2" color="text.secondary">
                or
              </Typography>
            </Divider>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, width: '100%' }}>
              {showGithub && (
                <Button
                  variant="outlined"
                  size="large"
                  fullWidth
                  startIcon={<GitHubIcon />}
                  onClick={() => handleOAuthSignIn('github')}
                  sx={{
                    py: 1.3,
                    borderColor: 'divider',
                    color: 'text.primary',
                    '&:hover': { borderColor: 'text.secondary', bgcolor: 'action.hover' },
                  }}
                >
                  Sign in with GitHub
                </Button>
              )}
              {showGoogle && (
                <Button
                  variant="outlined"
                  size="large"
                  fullWidth
                  startIcon={<GoogleIcon />}
                  onClick={() => handleOAuthSignIn('google')}
                  sx={{
                    py: 1.3,
                    borderColor: 'divider',
                    color: 'text.primary',
                    '&:hover': { borderColor: 'text.secondary', bgcolor: 'action.hover' },
                  }}
                >
                  Sign in with Google
                </Button>
              )}
            </Box>
          </>
        )}
      </Box>
    </Box>
  );
}

export default function LoginPage() {
  const { mode } = useThemeMode();
  const theme = createAppTheme(mode);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Suspense>
        <LoginForm />
      </Suspense>
    </ThemeProvider>
  );
}
