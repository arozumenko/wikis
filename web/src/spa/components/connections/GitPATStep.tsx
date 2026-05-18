/**
 * GitPATStep — wizard step 2 for Git PAT configuration.
 *
 * Simple form: repo URL, branch, PAT, label.
 * On "Next" the caller receives the form values to persist in step 3 (Confirm).
 */
import { useState } from 'react';
import {
  Box,
  TextField,
  Typography,
} from '@mui/material';
import { repoUrlId } from '../../lib/git-pat';
import type { GitConnection } from '../../hooks/useConnections';

interface GitPATStepProps {
  /** Called when the form is valid and the user clicks Next in the wizard. */
  onReady: (connection: GitConnection) => void;
  /** Ref forwarded from the wizard so the Step wrapper's Next button can trigger submit. */
  submitRef: React.MutableRefObject<(() => void) | null>;
}

export function GitPATStep({ onReady, submitRef }: GitPATStepProps) {
  const [repoUrl, setRepoUrl] = useState('');
  const [branch, setBranch] = useState('main');
  const [pat, setPat] = useState('');
  const [label, setLabel] = useState('');
  const [errors, setErrors] = useState<Partial<Record<'repoUrl' | 'pat', string>>>({});

  function validate(): boolean {
    const next: typeof errors = {};
    if (!repoUrl.trim()) next.repoUrl = 'Repository URL is required.';
    if (!pat.trim()) next.pat = 'Personal access token is required.';
    setErrors(next);
    return Object.keys(next).length === 0;
  }

  async function handleSubmit() {
    if (!validate()) return;
    const id = await repoUrlId(repoUrl.trim());
    const connection: GitConnection = {
      provider: 'git',
      id,
      repo_url: repoUrl.trim(),
      branch: branch.trim() || 'main',
      pat: pat.trim(),
      label: label.trim(),
      created_at: Date.now(),
    };
    onReady(connection);
  }

  // Expose submit so the parent wizard's "Next" button can trigger it
  submitRef.current = handleSubmit;

  return (
    <Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Enter your Git repository details. The PAT is stored locally in this browser
        and never sent to the Wikis backend directly.
      </Typography>

      <TextField
        label="Repository URL"
        placeholder="https://github.com/org/repo"
        value={repoUrl}
        onChange={(e) => setRepoUrl(e.target.value)}
        error={!!errors.repoUrl}
        helperText={errors.repoUrl}
        required
        fullWidth
        margin="normal"
      />

      <TextField
        label="Branch"
        value={branch}
        onChange={(e) => setBranch(e.target.value)}
        fullWidth
        margin="normal"
        helperText="Defaults to main if left blank."
      />

      <TextField
        label="Personal Access Token"
        value={pat}
        onChange={(e) => setPat(e.target.value)}
        error={!!errors.pat}
        helperText={
          errors.pat ||
          'Generate a fine-grained token with read access to repository contents.'
        }
        required
        fullWidth
        margin="normal"
        type="password"
        autoComplete="new-password"
      />

      <TextField
        label="Label (optional)"
        placeholder="e.g. My private repo"
        value={label}
        onChange={(e) => setLabel(e.target.value)}
        fullWidth
        margin="normal"
        helperText="A friendly name shown in the connections list."
      />
    </Box>
  );
}
