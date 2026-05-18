/**
 * AtlassianConnect — inline connect button + PKCE popup flow.
 *
 * Extracted from the deleted AtlassianStep wizard step so it can be
 * embedded directly inside ConfluenceConfigure and JiraConfigure.
 * When the OAuth flow completes, calls onConnected() which triggers
 * the parent to re-render once useConnections().atlassian is populated.
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
import { startAtlassianOAuth } from '../../../lib/atlassian-oauth';

interface AtlassianConnectProps {
  /** Called when the OAuth popup posts a success message. */
  onConnected: () => void;
}

type Phase = 'idle' | 'waiting' | 'success' | 'error';

export function AtlassianConnect({ onConnected }: AtlassianConnectProps) {
  const [phase, setPhase] = useState<Phase>('idle');
  const [error, setError] = useState('');
  const listenerAttached = useRef(false);

  // Attach the postMessage listener once per mount
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
        // Brief delay so the success indicator is visible
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
      // Popup is open; wait for the postMessage
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
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 1,
          }}
        >
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
        <Box
          sx={{ display: 'flex', alignItems: 'center', gap: 1, color: 'success.main' }}
        >
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
          <Button
            variant="contained"
            size="small"
            onClick={() => void handleConnect()}
          >
            Try Again
          </Button>
        </Box>
      )}
    </Box>
  );
}
