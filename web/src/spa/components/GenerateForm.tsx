import { useCallback, useState } from 'react';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import {
  Box,
  Button,
  FormControlLabel,
  Stack,
  Switch,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from '@mui/material';
import type { components } from '../api/types.generated';

type GenerateWikiRequest = components['schemas']['GenerateWikiRequest'];
type PlannerType = NonNullable<GenerateWikiRequest['planner_type']>;

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

const PLANNER_OPTIONS: ReadonlyArray<{
  value: PlannerType;
  label: string;
  shortHint: string;
  description: string;
}> = [
  {
    value: 'agent',
    label: 'Agentic',
    shortHint: 'LLM-driven outline',
    description:
      'An LLM agent explores the repository and decides the wiki outline. Slower and uses more tokens, but adapts coverage to what the model finds important. Test inclusion is decided automatically.',
  },
  {
    value: 'cluster',
    label: 'Graph clustering',
    shortHint: 'Leiden · fast · deterministic',
    description:
      'Builds a code graph, runs Leiden clustering, and turns each cluster into a wiki section. Faster and deterministic. Pairs well with “smart skip tests” to focus the outline on production code.',
  },
];

export function GenerateForm({ onSubmit, disabled = false, initialUrl = '' }: GenerateFormProps) {
  const [repoUrl, setRepoUrl] = useState(initialUrl);
  const [branch, setBranch] = useState('main');
  const [accessToken, setAccessToken] = useState('');
  const [plannerType, setPlannerType] = useState<PlannerType>('agent');
  const [excludeTests, setExcludeTests] = useState(true);

  const provider = detectProvider(repoUrl);
  const isLocal = provider === 'local';
  const isCluster = plannerType === 'cluster';

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
        // Server ignores exclude_tests for the agent planner.
        exclude_tests: isCluster ? excludeTests : null,
      });
    },
    [repoUrl, branch, provider, isLocal, accessToken, plannerType, isCluster, excludeTests, onSubmit],
  );

  const styles = generateFormStyles();

  return (
    <Box component="form" onSubmit={handleSubmit} sx={styles.root}>
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

      <Box sx={styles.plannerSection}>
        <Stack direction="row" alignItems="center" spacing={0.75} sx={styles.plannerHeader}>
          <Typography component="span" sx={styles.plannerLabel}>
            Structure planner
          </Typography>
          <Tooltip
            arrow
            placement="top"
            title={
              <Stack spacing={1} sx={{ p: 0.5 }}>
                {PLANNER_OPTIONS.map((opt) => (
                  <Box key={opt.value}>
                    <Typography variant="caption" sx={{ fontWeight: 700, display: 'block' }}>
                      {opt.label}
                    </Typography>
                    <Typography variant="caption">{opt.description}</Typography>
                  </Box>
                ))}
              </Stack>
            }
          >
            <InfoOutlinedIcon sx={styles.plannerInfoIcon} />
          </Tooltip>
        </Stack>

        <ToggleButtonGroup
          exclusive
          fullWidth
          size="small"
          value={plannerType}
          onChange={(_e, next) => {
            if (next) setPlannerType(next as PlannerType);
          }}
          disabled={disabled}
          aria-label="structure planner"
          sx={styles.plannerToggle}
        >
          {PLANNER_OPTIONS.map((option) => (
            <ToggleButton key={option.value} value={option.value} sx={styles.plannerToggleBtn}>
              <Stack spacing={0} alignItems="center">
                <Typography component="span" sx={styles.plannerToggleLabel}>
                  {option.label}
                </Typography>
                <Typography component="span" sx={styles.plannerToggleHint}>
                  {option.shortHint}
                </Typography>
              </Stack>
            </ToggleButton>
          ))}
        </ToggleButtonGroup>

        {isCluster && (
          <Box sx={styles.skipTestsRow}>
            <FormControlLabel
              sx={styles.skipTestsControl}
              control={
                <Switch
                  size="small"
                  checked={excludeTests}
                  onChange={(e) => setExcludeTests(e.target.checked)}
                  disabled={disabled}
                />
              }
              label={
                <Stack direction="row" alignItems="center" spacing={0.5}>
                  <Typography component="span" sx={styles.skipTestsLabel}>
                    Smart skip tests
                  </Typography>
                  <Tooltip
                    arrow
                    placement="top"
                    title="Drop test files and test-only nodes from the code graph before clustering so the wiki focuses on production code."
                  >
                    <InfoOutlinedIcon sx={styles.plannerInfoIcon} />
                  </Tooltip>
                </Stack>
              }
            />
          </Box>
        )}
      </Box>

      <Button
        type="submit"
        variant="contained"
        size="large"
        fullWidth
        sx={styles.submit}
        disabled={disabled || !repoUrl}
      >
        Generate Wiki
      </Button>
    </Box>
  );
}

const generateFormStyles = () => ({
  root: {
    maxWidth: '37.5rem', // 600px
    mx: 'auto',
  },
  plannerSection: {
    mt: 2,
    width: '100%',
  },
  plannerHeader: {
    mb: 0.75,
  },
  plannerLabel: ({ palette }: { palette: { text: { secondary: string } } }) => ({
    fontSize: '0.75rem',
    fontWeight: 600,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.04em',
    color: palette.text.secondary,
  }),
  plannerInfoIcon: ({ palette }: { palette: { text: { secondary: string } } }) => ({
    fontSize: '1rem',
    color: palette.text.secondary,
    cursor: 'help',
  }),
  plannerToggle: {
    '& .MuiToggleButton-root': {
      textTransform: 'none' as const,
      py: 1,
      px: 1.5,
    },
  },
  plannerToggleBtn: {
    flex: 1,
  },
  plannerToggleLabel: {
    fontSize: '0.875rem',
    fontWeight: 600,
    lineHeight: 1.2,
  },
  plannerToggleHint: ({ palette }: { palette: { text: { secondary: string } } }) => ({
    fontSize: '0.6875rem',
    color: palette.text.secondary,
    lineHeight: 1.2,
    mt: '0.125rem',
  }),
  skipTestsRow: {
    mt: 1,
    display: 'flex',
    justifyContent: 'flex-start',
  },
  skipTestsControl: {
    ml: 0,
    '& .MuiFormControlLabel-label': {
      ml: 0.5,
    },
  },
  skipTestsLabel: {
    fontSize: '0.875rem',
    fontWeight: 500,
  },
  submit: {
    mt: 2.5,
  },
});
