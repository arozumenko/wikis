/**
 * Git connector form for the wizard's Configure step (#208).
 *
 * Same field semantics as the old GenerateForm's GitTab (repo URL, branch,
 * PAT source select). Lifted into the wizard with the shared form-state
 * shape from ``../types.ts`` so the Confirm step can replay user input
 * without re-fetching anything.
 */

import { Alert, Box, FormControl, InputLabel, MenuItem, Select, TextField } from '@mui/material';
import { useConnections } from '../../../hooks/useConnections';
import type { GitFormState } from '../types';

interface GitConfigureProps {
  data: GitFormState;
  onChange: (next: GitFormState) => void;
  urlError: string | null;
  disabled?: boolean;
}

export function GitConfigure({ data, onChange, urlError, disabled }: GitConfigureProps) {
  const { connections } = useConnections();
  const gitConnections = connections.filter((c) => c.provider === 'git');

  const set = <K extends keyof GitFormState>(key: K, value: GitFormState[K]) =>
    onChange({ ...data, [key]: value });

  return (
    <Box>
      <TextField
        label="Repository URL"
        placeholder="https://github.com/owner/repo  ·  /abs/path for local"
        value={data.repo_url}
        onChange={(e) => set('repo_url', e.target.value)}
        required
        fullWidth
        margin="normal"
        error={!!urlError}
        helperText={urlError ?? undefined}
        disabled={disabled}
        inputProps={{ 'data-testid': 'git-repo-url' }}
      />

      <TextField
        label="Branch"
        value={data.branch}
        onChange={(e) => set('branch', e.target.value)}
        fullWidth
        margin="normal"
        disabled={disabled}
        inputProps={{ 'data-testid': 'git-branch' }}
      />

      <FormControl fullWidth margin="normal" disabled={disabled}>
        <InputLabel id="pat-source-label">Authentication</InputLabel>
        <Select
          labelId="pat-source-label"
          value={data.patSource}
          label="Authentication"
          onChange={(e) => set('patSource', e.target.value as GitFormState['patSource'])}
          inputProps={{ 'data-testid': 'pat-source-select' }}
        >
          <MenuItem value="none">No auth (public repo)</MenuItem>
          <MenuItem value="stored">Use stored PAT</MenuItem>
          <MenuItem value="paste">Paste token once (not stored)</MenuItem>
        </Select>
      </FormControl>

      {data.patSource === 'stored' && gitConnections.length === 0 && (
        <Alert severity="info" sx={{ mt: 1 }} data-testid="git-no-pat-alert">
          No stored Git PATs found. Switch to &ldquo;Paste token once&rdquo; above to
          provide a token without storing it.
        </Alert>
      )}

      {data.patSource === 'stored' && gitConnections.length > 0 && (
        <FormControl fullWidth margin="normal" disabled={disabled}>
          <InputLabel id="pat-select-label">Stored PAT</InputLabel>
          <Select
            labelId="pat-select-label"
            value={data.selectedPatId}
            label="Stored PAT"
            onChange={(e) => set('selectedPatId', e.target.value)}
            inputProps={{ 'data-testid': 'stored-pat-select' }}
          >
            {gitConnections.map((c) => (
              <MenuItem key={c.id} value={c.id}>
                {c.label || c.repo_url}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      )}

      {data.patSource === 'paste' && (
        <TextField
          label="Personal Access Token"
          value={data.pastedPat}
          onChange={(e) => set('pastedPat', e.target.value)}
          fullWidth
          margin="normal"
          type="password"
          disabled={disabled}
          helperText="Token will not be stored"
          inputProps={{ 'data-testid': 'pasted-pat-input' }}
        />
      )}
    </Box>
  );
}
