/**
 * Jira connector form for the wizard's Configure step (#208).
 *
 * Same shape as ``ConfluenceConfigure`` — reads the Atlassian connection
 * from ``useConnections``, shows a connect-link affordance when absent.
 */

import { Alert, Box, TextField, Typography } from '@mui/material';
import { Link } from 'react-router-dom';
import { useConnections } from '../../../hooks/useConnections';
import type { JiraFormState } from '../types';

interface JiraConfigureProps {
  data: JiraFormState;
  onChange: (next: JiraFormState) => void;
  jqlError: string | null;
  disabled?: boolean;
}

export function JiraConfigure({ data, onChange, jqlError, disabled }: JiraConfigureProps) {
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
    </Box>
  );
}
