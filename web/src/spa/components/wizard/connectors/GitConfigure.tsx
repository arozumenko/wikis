/**
 * Git connector form for the wizard's Configure step.
 *
 * Same field semantics as the old GenerateForm's GitTab (repo URL, branch,
 * PAT source select). Lifted into the wizard with the shared form-state
 * shape from ``../types.ts`` so the Confirm step can replay user input
 * without re-fetching anything.
 */

import { Box, FormControl, InputLabel, MenuItem, Select, TextField } from '@mui/material';
import type { GitFormState } from '../types';

interface GitConfigureProps {
  data: GitFormState;
  onChange: (next: GitFormState) => void;
  urlError: string | null;
  disabled?: boolean;
}

export function GitConfigure({ data, onChange, urlError, disabled }: GitConfigureProps) {
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
          <MenuItem value="paste">Paste token once (not stored)</MenuItem>
        </Select>
      </FormControl>

      {data.patSource === 'paste' && (
        <TextField
          label="Personal Access Token"
          value={data.pastedPat}
          onChange={(e) => set('pastedPat', e.target.value)}
          required
          fullWidth
          margin="normal"
          type="password"
          disabled={disabled}
          error={!data.pastedPat.trim()}
          helperText={
            !data.pastedPat.trim()
              ? 'Token is required for the "Paste token once" option'
              : 'Token will not be stored'
          }
          inputProps={{ 'data-testid': 'pasted-pat-input' }}
        />
      )}
    </Box>
  );
}
