/**
 * Jira connector form for the wizard's Configure step.
 *
 * Two auth modes are supported (same as ConfluenceConfigure):
 * - OAuth: reads from ``useConnections().atlassian``.
 * - API token (email + token): credentials live in the wizard form state.
 */

import { Box, TextField, Typography } from '@mui/material';
import { useConnections } from '../../../hooks/useConnections';
import { AtlassianConnect } from './AtlassianConnect';
import type {
  AtlassianAuthMode,
  AtlassianBasicAuthFormState,
  JiraFormState,
} from '../types';

interface JiraConfigureProps {
  data: JiraFormState;
  onChange: (next: JiraFormState) => void;
  jqlError: string | null;
  disabled?: boolean;
  authMode: AtlassianAuthMode;
  onAuthModeChange: (mode: AtlassianAuthMode) => void;
  basicAuth: AtlassianBasicAuthFormState;
  onBasicAuthChange: (next: AtlassianBasicAuthFormState) => void;
}

export function JiraConfigure({
  data,
  onChange,
  jqlError,
  disabled,
  authMode,
  onAuthModeChange,
  basicAuth,
  onBasicAuthChange,
}: JiraConfigureProps) {
  const { atlassian, saveAtlassian } = useConnections();

  let baseUrl: string | null = null;
  let siteLabel: string | null = null;
  if (authMode === 'oauth' && atlassian) {
    baseUrl = atlassian.accessible_resources[0]?.url ?? atlassian.site_name;
    siteLabel = atlassian.site_name;
  } else if (authMode === 'api_token' && basicAuth.siteUrl.trim()) {
    baseUrl = basicAuth.siteUrl.trim();
    siteLabel = baseUrl;
  }

  const ready =
    authMode === 'oauth'
      ? Boolean(atlassian)
      : Boolean(basicAuth.siteUrl.trim() && basicAuth.email.trim() && basicAuth.apiToken);

  return (
    <Box>
      <AtlassianConnect
        mode={authMode}
        onModeChange={onAuthModeChange}
        basic={basicAuth}
        onBasicChange={onBasicAuthChange}
        onConnected={() => {
          const stored = localStorage.getItem('wikis.connections.atlassian');
          if (stored) {
            try {
              saveAtlassian(JSON.parse(stored));
            } catch {
              // ignore malformed entry
            }
          }
        }}
      />

      {ready && baseUrl && (
        <>
          <Box sx={{ mt: 2, mb: 1 }}>
            <Typography variant="caption" color="text.secondary">
              Atlassian site
            </Typography>
            {siteLabel && (
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {siteLabel}
              </Typography>
            )}
            <Typography variant="caption" color="text.secondary">
              {baseUrl}
            </Typography>
          </Box>

          <TextField
            label="JQL Query"
            value={data.jql}
            onChange={(e) => onChange({ jql: e.target.value })}
            fullWidth
            margin="normal"
            multiline
            minRows={2}
            error={!!jqlError}
            helperText={jqlError ?? 'Jira Query Language filter for issues to include'}
            disabled={disabled}
            inputProps={{ 'data-testid': 'jql-input' }}
          />
        </>
      )}
    </Box>
  );
}
