/**
 * OAuthCallback — handles the Atlassian OAuth 2.0 PKCE callback.
 *
 * Mounted at /oauth/atlassian/callback inside the SPA's BrowserRouter.
 * This is NOT a Next.js App Router page — it lives inside the catch-all SPA
 * route. The middleware does not proxy /oauth/* paths.
 *
 * Flow:
 *   1. Read `code` + `state` from URL search params
 *   2. Compare `state` against sessionStorage value (CSRF guard)
 *   3. Exchange `code` for tokens via Atlassian's token endpoint
 *   4. Discover accessible cloud sites
 *   5. Persist via useConnections.saveAtlassian
 *   6. postMessage to opener so the wizard knows the flow is done
 *   7. Close the popup
 */
import { useEffect, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Box, CircularProgress, Typography } from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import {
  completeAtlassianOAuth,
  fetchAccessibleResources,
} from '../lib/atlassian-oauth';
import { useConnections } from '../hooks/useConnections';
import type { AtlassianConnection } from '../hooks/useConnections';

type Phase = 'exchanging' | 'success' | 'error';

export function OAuthCallback() {
  const [searchParams] = useSearchParams();
  const { saveAtlassian } = useConnections();
  const [phase, setPhase] = useState<Phase>('exchanging');
  const [errorMsg, setErrorMsg] = useState('');
  const ran = useRef(false); // Prevent double-execution in StrictMode

  useEffect(() => {
    if (ran.current) return;
    ran.current = true;

    const code = searchParams.get('code');
    const state = searchParams.get('state');

    async function finish() {
      if (!code || !state) {
        setErrorMsg('Missing code or state parameter.');
        setPhase('error');
        return;
      }

      try {
        const tokens = await completeAtlassianOAuth(code, state);
        const resources = await fetchAccessibleResources(tokens.access_token);

        // Use the first accessible resource as the primary site
        const primary = resources[0];
        const connection: AtlassianConnection = {
          provider: 'atlassian',
          access_token: tokens.access_token,
          refresh_token: tokens.refresh_token,
          expires_at: tokens.expires_at,
          cloud_id: primary?.id ?? '',
          site_name: primary?.name ?? '',
          accessible_resources: resources,
          created_at: Date.now(),
        };

        saveAtlassian(connection);
        setPhase('success');

        // Notify the opener (the wizard) and close
        if (window.opener) {
          window.opener.postMessage(
            { type: 'wikis-oauth-success', provider: 'atlassian' },
            window.location.origin,
          );
        }

        // Brief pause so the user sees the success state before the popup closes
        setTimeout(() => window.close(), 1200);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Unexpected error during OAuth flow.';
        setErrorMsg(message);
        setPhase('error');
      }
    }

    void finish();
  }, [searchParams, saveAtlassian]);

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        gap: 2,
        px: 3,
        textAlign: 'center',
      }}
    >
      {phase === 'exchanging' && (
        <>
          <CircularProgress />
          <Typography variant="body1" color="text.secondary">
            Connecting to Atlassian...
          </Typography>
        </>
      )}

      {phase === 'success' && (
        <>
          <CheckCircleOutlineIcon
            sx={{ fontSize: 56, color: 'success.main' }}
          />
          <Typography variant="h6">Connected!</Typography>
          <Typography variant="body2" color="text.secondary">
            This window will close automatically.
          </Typography>
        </>
      )}

      {phase === 'error' && (
        <>
          <ErrorOutlineIcon sx={{ fontSize: 56, color: 'error.main' }} />
          <Typography variant="h6">Connection failed</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 360 }}>
            {errorMsg}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            You can close this window and try again.
          </Typography>
        </>
      )}
    </Box>
  );
}
