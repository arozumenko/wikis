/**
 * Confluence connector form for the wizard's Configure step (#208).
 *
 * Reads the Atlassian connection from ``useConnections`` (localStorage-
 * backed). If absent, shows a connect-link instead of the form fields —
 * #211 will replace this with an inline connect affordance once Atlassian
 * scan endpoints land.
 */

import { Alert, Autocomplete, Box, Chip, TextField, Typography } from '@mui/material';
import { Link } from 'react-router-dom';
import { useConnections } from '../../../hooks/useConnections';
import type { ConfluenceFormState } from '../types';

interface ConfluenceConfigureProps {
  data: ConfluenceFormState;
  onChange: (next: ConfluenceFormState) => void;
  spaceKeysError: string | null;
  disabled?: boolean;
}

export function ConfluenceConfigure({
  data,
  onChange,
  spaceKeysError,
  disabled,
}: ConfluenceConfigureProps) {
  const { atlassian } = useConnections();

  if (!atlassian) {
    return (
      <Box sx={{ py: 2 }}>
        <Alert severity="warning" data-testid="atlassian-connect-warning">
          No Atlassian connection found.{' '}
          <Link to="/settings?tab=connections" style={{ color: 'inherit' }}>
            Connect to Atlassian in Settings
          </Link>
          .
        </Alert>
      </Box>
    );
  }

  const baseUrl = atlassian.accessible_resources[0]?.url ?? atlassian.site_name;

  return (
    <Box>
      <Box sx={{ mt: 1, mb: 1 }}>
        <Typography variant="caption" color="text.secondary">
          Atlassian site
        </Typography>
        <Typography variant="body2" sx={{ fontWeight: 500 }}>
          {atlassian.site_name}
        </Typography>
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
          // Handle comma-separated paste — split, trim, dedupe.
          if (reason === 'input' && value.includes(',')) {
            const incoming = value.split(',').map((k) => k.trim()).filter(Boolean);
            const merged = [...new Set([...data.space_keys, ...incoming])];
            onChange({ space_keys: merged });
          }
        }}
      />
    </Box>
  );
}
