import { useState, useCallback, useEffect } from 'react';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  TextField,
  Typography,
} from '@mui/material';

const AUTH_URL = ''; // Same origin

export function AccountTab() {
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [email, setEmail] = useState<string | null>(null);
  const [hasPassword, setHasPassword] = useState<boolean | null>(null);

  useEffect(() => {
    // Fetch session for email display
    fetch(`${AUTH_URL}/api/auth/get-session`, { credentials: 'include' })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data?.user?.email) setEmail(data.user.email);
      })
      .catch(() => {});

    // Fetch linked accounts to check if user has a password (credential) account
    fetch(`${AUTH_URL}/api/auth/list-accounts`, {
      method: 'GET',
      credentials: 'include',
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        const accounts: { providerId: string }[] = data ?? [];
        setHasPassword(accounts.some((a) => a.providerId === 'credential'));
      })
      .catch(() => {
        // If the endpoint isn't available, assume credential account (safe default)
        setHasPassword(true);
      });
  }, []);

  const handleSubmit = useCallback(async () => {
    setMessage(null);

    if (newPassword.length < 8) {
      setMessage({ type: 'error', text: 'New password must be at least 8 characters.' });
      return;
    }
    if (newPassword !== confirmPassword) {
      setMessage({ type: 'error', text: 'New passwords do not match.' });
      return;
    }

    setLoading(true);
    try {
      const resp = await fetch(`${AUTH_URL}/api/auth/change-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          currentPassword,
          newPassword,
        }),
      });

      if (resp.ok) {
        setMessage({ type: 'success', text: 'Password changed successfully.' });
        setCurrentPassword('');
        setNewPassword('');
        setConfirmPassword('');
      } else {
        const data = await resp.json().catch(() => null);
        setMessage({
          type: 'error',
          text: data?.message ?? 'Failed to change password. Check your current password.',
        });
      }
    } catch {
      setMessage({ type: 'error', text: 'Network error. Please try again.' });
    }
    setLoading(false);
  }, [currentPassword, newPassword, confirmPassword]);

  return (
    <Box>
      <Typography variant="h6" sx={{ mb: 1 }}>
        Account
      </Typography>

      {email && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Signed in as <strong>{email}</strong>
        </Typography>
      )}

      {hasPassword === false ? (
        <Box>
          <Typography variant="subtitle1" sx={{ mb: 1 }}>
            Password
          </Typography>
          <Alert severity="info">
            Your account uses external authentication (GitHub, Google, etc.).
            Password management is handled by your identity provider.
          </Alert>
        </Box>
      ) : (
        <>
          <Typography variant="subtitle1" sx={{ mb: 2 }}>
            Change Password
          </Typography>

          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, maxWidth: 400 }}>
            <TextField
              label="Current password"
              type="password"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              autoComplete="current-password"
              size="small"
            />
            <TextField
              label="New password"
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              autoComplete="new-password"
              size="small"
              helperText="Minimum 8 characters"
            />
            <TextField
              label="Confirm new password"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              autoComplete="new-password"
              size="small"
              error={confirmPassword.length > 0 && newPassword !== confirmPassword}
              helperText={
                confirmPassword.length > 0 && newPassword !== confirmPassword
                  ? 'Passwords do not match'
                  : undefined
              }
            />

            {message && (
              <Alert severity={message.type} onClose={() => setMessage(null)}>
                {message.text}
              </Alert>
            )}

            <Button
              variant="contained"
              onClick={handleSubmit}
              disabled={loading || !currentPassword || !newPassword || !confirmPassword}
              sx={{ alignSelf: 'flex-start' }}
            >
              {loading ? <CircularProgress size={20} /> : 'Change Password'}
            </Button>
          </Box>
        </>
      )}
    </Box>
  );
}
