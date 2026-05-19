/**
 * AtlassianConnect — connect surface for Atlassian (Confluence / Jira).
 *
 * Renders two tabs:
 *
 * 1. **OAuth** — kicks off the in-app PKCE popup flow.  When the popup posts
 *    a success message, ``onConnected`` fires and the parent re-renders
 *    once ``useConnections().atlassian`` is populated.
 * 2. **API token** — collects site URL + email + API token (the shape used
 *    by ``atlassian-python-api``).  Credentials live in the wizard form
 *    state (``atlassianBasic`` slice) and are NOT persisted anywhere — the
 *    user re-enters them per session.
 *
 * The mode the user selected is reflected via the ``mode`` /
 * ``onModeChange`` props so the parent can branch its scan / submit
 * payload accordingly.
 */
import { useEffect, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Link,
  Stack,
  Tab,
  Tabs,
  TextField,
  Typography,
} from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import { startAtlassianOAuth } from '../../../lib/atlassian-oauth';
import type {
  AtlassianAuthMode,
  AtlassianBasicAuthFormState,
} from '../types';

interface AtlassianConnectProps {
  mode: AtlassianAuthMode;
  onModeChange: (mode: AtlassianAuthMode) => void;
  basic: AtlassianBasicAuthFormState;
  onBasicChange: (next: AtlassianBasicAuthFormState) => void;
  /** Called when the OAuth popup posts a success message. */
  onConnected: () => void;
}

type OAuthPhase = 'idle' | 'waiting' | 'success' | 'error';

export function AtlassianConnect({
  mode,
  onModeChange,
  basic,
  onBasicChange,
  onConnected,
}: AtlassianConnectProps) {
  return (
    <Box sx={{ width: '100%' }}>
      <Tabs
        value={mode}
        onChange={(_e, val) => onModeChange(val as AtlassianAuthMode)}
        sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}
        aria-label="atlassian authentication mode"
      >
        <Tab
          value="oauth"
          label="OAuth"
          data-testid="atlassian-tab-oauth"
        />
        <Tab
          value="api_token"
          label="API token"
          data-testid="atlassian-tab-api-token"
        />
      </Tabs>

      {mode === 'oauth' && <OAuthPane onConnected={onConnected} />}
      {mode === 'api_token' && (
        <ApiTokenPane basic={basic} onBasicChange={onBasicChange} />
      )}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// OAuth pane (preserves the prior behaviour verbatim)
// ---------------------------------------------------------------------------

function OAuthPane({ onConnected }: { onConnected: () => void }) {
  const [phase, setPhase] = useState<OAuthPhase>('idle');
  const [error, setError] = useState('');
  const listenerAttached = useRef(false);

  useEffect(() => {
    if (listenerAttached.current) return;
    listenerAttached.current = true;

    function onMessage(event: MessageEvent) {
      if (event.origin !== window.location.origin) return;
      if (
        event.data?.type === 'wikis-oauth-success' &&
        event.data?.provider === 'atlassian'
      ) {
        setPhase('success');
        setTimeout(onConnected, 600);
      }
    }

    window.addEventListener('message', onMessage);
    return () => {
      window.removeEventListener('message', onMessage);
      listenerAttached.current = false;
    };
  }, [onConnected]);

  async function handleConnect() {
    setError('');
    setPhase('waiting');
    try {
      await startAtlassianOAuth();
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : 'Failed to start Atlassian OAuth.';
      setError(msg);
      setPhase('error');
    }
  }

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2,
        py: 1,
      }}
    >
      {phase === 'idle' && (
        <Button
          variant="contained"
          onClick={() => void handleConnect()}
          data-testid="atlassian-connect-button"
        >
          Connect to Atlassian
        </Button>
      )}

      {phase === 'waiting' && (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
          <CircularProgress size={24} />
          <Typography variant="body2" color="text.secondary">
            Waiting for authorization in the popup window…
          </Typography>
          <Button variant="text" size="small" onClick={() => setPhase('idle')}>
            Cancel
          </Button>
        </Box>
      )}

      {phase === 'success' && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: 'success.main' }}>
          <CheckCircleOutlineIcon fontSize="small" />
          <Typography variant="body2" color="success.main">
            Connected successfully!
          </Typography>
        </Box>
      )}

      {phase === 'error' && (
        <Box sx={{ width: '100%' }}>
          <Alert severity="error" sx={{ mb: 1 }}>
            {error}
          </Alert>
          <Button variant="contained" size="small" onClick={() => void handleConnect()}>
            Try Again
          </Button>
        </Box>
      )}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// API-token pane
// ---------------------------------------------------------------------------

function ApiTokenPane({
  basic,
  onBasicChange,
}: {
  basic: AtlassianBasicAuthFormState;
  onBasicChange: (next: AtlassianBasicAuthFormState) => void;
}) {
  const set = <K extends keyof AtlassianBasicAuthFormState>(
    key: K,
    value: AtlassianBasicAuthFormState[K],
  ) => onBasicChange({ ...basic, [key]: value });

  return (
    <Stack spacing={1.5} sx={{ pt: 1 }}>
      <Typography variant="caption" color="text.secondary">
        Use this with an Atlassian API token (compatible with the{' '}
        <Link
          href="https://atlassian-python-api.readthedocs.io/"
          target="_blank"
          rel="noopener noreferrer"
          underline="hover"
        >
          atlassian-python-api
        </Link>
        {' '}auth shape).  Generate a token at{' '}
        <Link
          href="https://id.atlassian.com/manage-profile/security/api-tokens"
          target="_blank"
          rel="noopener noreferrer"
          underline="hover"
        >
          id.atlassian.com/manage-profile/security/api-tokens
        </Link>
        .  Credentials live in this session only and are never stored on the server.
      </Typography>

      <TextField
        label="Site URL"
        placeholder="https://your-tenant.atlassian.net"
        value={basic.siteUrl}
        onChange={(e) => set('siteUrl', e.target.value)}
        required
        fullWidth
        size="small"
        inputProps={{ 'data-testid': 'atlassian-basic-site-url' }}
      />

      <TextField
        label="Email"
        placeholder="you@example.com"
        type="email"
        value={basic.email}
        onChange={(e) => set('email', e.target.value)}
        required
        fullWidth
        size="small"
        inputProps={{ 'data-testid': 'atlassian-basic-email' }}
      />

      <TextField
        label="API token"
        type="password"
        value={basic.apiToken}
        onChange={(e) => set('apiToken', e.target.value)}
        required
        fullWidth
        size="small"
        helperText="Token will not be stored"
        inputProps={{ 'data-testid': 'atlassian-basic-api-token' }}
      />
    </Stack>
  );
}
