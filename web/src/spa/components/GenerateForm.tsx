import { useCallback, useState } from 'react';
import {
  Box,
  Button,
  Checkbox,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Select,
  TextField,
} from '@mui/material';
import type { components } from '../api/types.generated';

type GenerateWikiRequest = components['schemas']['GenerateWikiRequest'];

interface GenerateFormProps {
  onSubmit: (request: GenerateWikiRequest) => void;
  disabled?: boolean;
  initialUrl?: string;
}

function isLocalPath(url: string): boolean {
  return url.startsWith('/') || url.startsWith('file://');
}

function detectProvider(url: string): string {
  if (isLocalPath(url)) return 'local';
  if (url.includes('github.com')) return 'github';
  if (url.includes('gitlab.com') || url.includes('gitlab.')) return 'gitlab';
  if (url.includes('bitbucket.org')) return 'bitbucket';
  if (url.includes('dev.azure.com') || url.includes('visualstudio.com')) return 'ado';
  return 'github';
}

export function GenerateForm({ onSubmit, disabled = false, initialUrl = '' }: GenerateFormProps) {
  const [repoUrl, setRepoUrl] = useState(initialUrl);
  const [branch, setBranch] = useState('main');
  const [accessToken, setAccessToken] = useState('');
  const [plannerType, setPlannerType] = useState<'agent' | 'cluster'>('agent');
  const [excludeTests, setExcludeTests] = useState(true);

  const provider = detectProvider(repoUrl);
  const isLocal = provider === 'local';

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      onSubmit({
        repo_url: repoUrl,
        branch,
        provider,
        access_token: isLocal ? null : accessToken || null,
        wiki_title: null,
        include_research: true,
        include_diagrams: true,
        force_rebuild_index: false,
        llm_model: null,
        embedding_model: null,
        visibility: 'personal',
        planner_type: plannerType,
        exclude_tests: excludeTests,
      });
    },
    [repoUrl, branch, provider, isLocal, accessToken, plannerType, excludeTests, onSubmit],
  );

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ maxWidth: 600, mx: 'auto' }}>
      <TextField
        label="Repository URL or Local Path"
        placeholder="https://github.com/user/repo or /home/user/my-project"
        value={repoUrl}
        onChange={(e) => setRepoUrl(e.target.value)}
        required
        fullWidth
        margin="normal"
        helperText={
          isLocal
            ? 'Local path — ensure it is listed in ALLOWED_LOCAL_PATHS on the server'
            : repoUrl
              ? `Detected provider: ${provider}`
              : undefined
        }
        disabled={disabled}
      />

      <TextField
        label="Branch"
        value={branch}
        onChange={(e) => setBranch(e.target.value)}
        fullWidth
        margin="normal"
        helperText={isLocal ? 'Leave blank for non-git directories' : undefined}
        disabled={disabled}
      />

      {!isLocal && (
        <TextField
          label="Access Token (optional, for private repos)"
          value={accessToken}
          onChange={(e) => setAccessToken(e.target.value)}
          fullWidth
          margin="normal"
          type="password"
          disabled={disabled}
        />
      )}

      <FormControl fullWidth margin="normal" disabled={disabled}>
        <InputLabel id="planner-type-label">Structure Planner</InputLabel>
        <Select
          labelId="planner-type-label"
          value={plannerType}
          label="Structure Planner"
          onChange={(e) => setPlannerType(e.target.value as 'agent' | 'cluster')}
        >
          <MenuItem value="agent">Agent (LLM-guided)</MenuItem>
          <MenuItem value="cluster">Cluster (graph-based)</MenuItem>
        </Select>
      </FormControl>

      <FormControlLabel
        control={
          <Checkbox
            checked={excludeTests}
            onChange={(e) => setExcludeTests(e.target.checked)}
            disabled={disabled}
          />
        }
        label="Exclude test files from wiki structure"
        sx={{ mt: 1 }}
      />

      <Button
        type="submit"
        variant="contained"
        size="large"
        fullWidth
        sx={{ mt: 3 }}
        disabled={disabled || !repoUrl}
      >
        Generate Wiki
      </Button>
    </Box>
  );
}
