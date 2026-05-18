/**
 * AtlassianStep — wizard step 2 for Atlassian OAuth PKCE.
 *
 * Shows a "Connect to Atlassian" button. When clicked it:
 *   1. Calls startAtlassianOAuth() which opens a popup
 *   2. Shows a spinner while the popup is open
 *   3. Listens for a `wikis-oauth-success` postMessage from the callback page
 *   4. On success, calls onConnected() so the wizard advances to step 3
 */
import { useEffect, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Typography,
} from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import { startAtlassianOAuth } from '../../lib/atlassian-oauth';

interface AtlassianStepProps {
  /** Called when the OAuth flow completes successfully. */
  onConnected: () => void;
}

type Phase = 'idle' | 'waiting' | 'success' | 'error';

export function AtlassianStep({ onConnected }: AtlassianStepProps) {
  const [phase, setPhase] = useState<Phase>('idle');
  const [error, setError] = useState('');
  const listenerAttached = useRef(false);

  // Attach the postMessage listener once
  useEffect(() => {
    if (listenerAttached.current) return;
    listenerAttached.current = true;

    function onMessage(event: MessageEvent) {
      // Only accept messages from our own origin
      if (event.origin !== window.location.origin) return;
      if (event.data?.type === 'wikis-oauth-success' && event.data?.provider === 'atlassian') {
        setPhase('success');
        // Brief delay so the success state is visible before the wizard advances
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
      // Popup is now open; we wait for the postMessage
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : 'Failed to start Atlassian OAuth.';
      setError(msg);
      setPhase('error');
    }
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3, py: 2 }}>
      {/* Provider logo placeholder */}
      <Box
        sx={{
          width: 64,
          height: 64,
          borderRadius: 2,
          bgcolor: '#0052CC',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#fff',
          fontWeight: 700,
          fontSize: '1.25rem',
        }}
      >
        AT
      </Box>

      <Box sx={{ textAlign: 'center', maxWidth: 400 }}>
        <Typography variant="h6" gutterBottom>
          Connect Atlassian
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Grant read-only access to Confluence and Jira. Your credentials are stored
          locally in this browser and never sent to the Wikis backend.
        </Typography>
      </Box>

      {phase === 'idle' && (
        <Button variant="contained" size="large" onClick={handleConnect}>
          Connect to Atlassian
        </Button>
      )}

      {phase === 'waiting' && (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1.5 }}>
          <CircularProgress />
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
          <CheckCircleOutlineIcon />
          <Typography variant="body1" color="success.main">
            Connected successfully!
          </Typography>
        </Box>
      )}

      {phase === 'error' && (
        <Box sx={{ width: '100%', maxWidth: 400 }}>
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
          <Button variant="contained" onClick={handleConnect}>
            Try Again
          </Button>
        </Box>
      )}

      <Typography variant="caption" color="text.secondary" sx={{ textAlign: 'center', maxWidth: 380 }}>
        Scopes requested: read:confluence-content.all, read:confluence-space.summary,
        read:jira-work, read:jira-user, offline_access
      </Typography>
    </Box>
  );
}
