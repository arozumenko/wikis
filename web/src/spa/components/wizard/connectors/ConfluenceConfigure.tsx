/**
 * Confluence connector form for the wizard's Configure step.
 *
 * Two auth modes are now supported:
 * - OAuth: reads from ``useConnections().atlassian``.  Site URL / space keys
 *   form is shown ONLY when an OAuth session is present.
 * - API token (email + token): credentials live in the wizard form state.
 *   The space-keys form is shown as soon as a valid site URL is entered.
 */

import { Autocomplete, Box, Chip, TextField, Typography } from '@mui/material';
import { useConnections } from '../../../hooks/useConnections';
import { AtlassianConnect } from './AtlassianConnect';
import type {
  AtlassianAuthMode,
  AtlassianBasicAuthFormState,
  ConfluenceFormState,
} from '../types';

interface ConfluenceConfigureProps {
  data: ConfluenceFormState;
  onChange: (next: ConfluenceFormState) => void;
  spaceKeysError: string | null;
  disabled?: boolean;
  authMode: AtlassianAuthMode;
  onAuthModeChange: (mode: AtlassianAuthMode) => void;
  basicAuth: AtlassianBasicAuthFormState;
  onBasicAuthChange: (next: AtlassianBasicAuthFormState) => void;
}

export function ConfluenceConfigure({
  data,
  onChange,
  spaceKeysError,
  disabled,
  authMode,
  onAuthModeChange,
  basicAuth,
  onBasicAuthChange,
}: ConfluenceConfigureProps) {
  const { atlassian, saveAtlassian } = useConnections();

  // Determine the active site URL based on the chosen auth mode.
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
          // OAuth callback fires from a popup; bump the connections store
          // so this component re-renders once the token is in localStorage.
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

          <Autocomplete
            multiple
            freeSolo
            options={[]}
            value={data.space_keys}
            disabled={disabled}
            onChange={(_e, value) => onChange({ space_keys: value as string[] })}
            renderTags={(value, getTagProps) =>
              value.map((option, index) => (
                <Chip
                  variant="outlined"
                  label={option}
                  size="small"
                  {...getTagProps({ index })}
                  key={option}
                />
              ))
            }
            renderInput={(params) => (
              <TextField
                {...params}
                label="Space keys"
                placeholder="e.g. ENG, DOCS"
                margin="normal"
                error={!!spaceKeysError}
                helperText={
                  spaceKeysError ?? 'Type a key and press Enter, or paste comma-separated keys'
                }
                inputProps={{ ...params.inputProps, 'data-testid': 'space-keys-input' }}
              />
            )}
            onInputChange={(_e, value, reason) => {
              if (reason === 'input' && value.includes(',')) {
                const incoming = value.split(',').map((k) => k.trim()).filter(Boolean);
                const merged = [...new Set([...data.space_keys, ...incoming])];
                onChange({ space_keys: merged });
              }
            }}
          />
        </>
      )}
    </Box>
  );
}
