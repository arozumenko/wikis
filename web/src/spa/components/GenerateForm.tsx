import { useCallback, useMemo, useState } from 'react';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import {
  Alert,
  Autocomplete,
  Box,
  Button,
  Chip,
  FormControl,
  FormHelperText,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Tab,
  Tabs,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import { Link } from 'react-router-dom';
import { useConnections } from '../hooks/useConnections';
import type { AtlassianAuth, GenerateWikiMultiSourceRequest, WikiSourceType } from '../api/wiki';
import type { components } from '../api/types.generated';

// ---------------------------------------------------------------------------
// Legacy shape — kept for backward compat with GeneratePage (direct URL mode)
// ---------------------------------------------------------------------------
type GenerateWikiRequest = components['schemas']['GenerateWikiRequest'];

// ---------------------------------------------------------------------------
// Planner options
// ---------------------------------------------------------------------------

type PlannerMode = 'agentic' | 'graph_clustering';

const PLANNER_OPTIONS: ReadonlyArray<{
  value: PlannerMode;
  label: string;
  shortHint: string;
  description: string;
}> = [
  {
    value: 'agentic',
    label: 'Agentic',
    shortHint: 'LLM-driven outline',
    description:
      'An LLM agent explores the repository and decides the wiki outline. Slower and uses more tokens, but adapts coverage to what the model finds important.',
  },
  {
    value: 'graph_clustering',
    label: 'Graph clustering',
    shortHint: 'Leiden · fast · deterministic',
    description:
      'Builds a code graph, runs Leiden clustering, and turns each cluster into a wiki section. Faster and deterministic.',
  },
];

// ---------------------------------------------------------------------------
// PAT source options (Git tab)
// ---------------------------------------------------------------------------

type PatSource = 'stored' | 'paste' | 'none';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

/**
 * GenerateForm accepts one of two submit callbacks:
 *
 * - `onSubmit` (legacy): receives a `GenerateWikiRequest` (old shape).
 *   Used by GeneratePage for direct URL generation.
 * - `onSubmitMultiSource`: receives a `GenerateWikiMultiSourceRequest` (new
 *   multi-source shape). Used by DashboardPage's Generate Wiki dialog.
 *
 * If `onSubmitMultiSource` is provided the form renders the multi-source UI
 * (tabs for Git / Confluence / Jira). Otherwise it renders the legacy single-
 * source form to avoid breaking GeneratePage.
 */
interface GenerateFormBaseProps {
  disabled?: boolean;
  initialUrl?: string;
}

interface GenerateFormLegacyProps extends GenerateFormBaseProps {
  onSubmit: (request: GenerateWikiRequest) => void;
  onSubmitMultiSource?: never;
}

interface GenerateFormMultiSourceProps extends GenerateFormBaseProps {
  onSubmit?: never;
  onSubmitMultiSource: (request: GenerateWikiMultiSourceRequest) => Promise<void>;
}

type GenerateFormProps = GenerateFormLegacyProps | GenerateFormMultiSourceProps;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

function isUrlish(url: string): boolean {
  return (
    url.startsWith('https://') ||
    url.startsWith('http://') ||
    url.startsWith('git@') ||
    url.startsWith('file://') ||
    url.startsWith('/')
  );
}

// ---------------------------------------------------------------------------
// Legacy form (unchanged behaviour)
// ---------------------------------------------------------------------------

function LegacyGenerateForm({
  onSubmit,
  disabled = false,
  initialUrl = '',
}: GenerateFormLegacyProps) {
  const [repoUrl, setRepoUrl] = useState(initialUrl);
  const [branch, setBranch] = useState('main');
  const [accessToken, setAccessToken] = useState('');
  const [plannerType, setPlannerType] = useState<'agent' | 'cluster'>('agent');
  // excludeTests is passed to the legacy request shape; the toggle UI was removed
  // in the multi-source form, so the setter is intentionally unused here.
  const [excludeTests] = useState(true);

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
        exclude_tests: isCluster ? excludeTests : null,
      });
    },
    [repoUrl, branch, provider, isLocal, accessToken, plannerType, isCluster, excludeTests, onSubmit],
  );

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ maxWidth: '37.5rem', mx: 'auto' }}>
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

      <PlannerSection
        plannerMode={plannerType === 'agent' ? 'agentic' : 'graph_clustering'}
        onChange={(m) => setPlannerType(m === 'agentic' ? 'agent' : 'cluster')}
        disabled={disabled}
      />

      <Button
        type="submit"
        variant="contained"
        size="large"
        fullWidth
        sx={{ mt: 2.5 }}
        disabled={disabled || !repoUrl}
      >
        Generate Wiki
      </Button>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Planner section (shared between legacy and multi-source forms)
// ---------------------------------------------------------------------------

interface PlannerSectionProps {
  plannerMode: PlannerMode;
  onChange: (mode: PlannerMode) => void;
  disabled?: boolean;
}

function PlannerSection({ plannerMode, onChange, disabled }: PlannerSectionProps) {
  return (
    <Box sx={{ mt: 2, width: '100%' }}>
      <Stack direction="row" alignItems="center" spacing={0.75} sx={{ mb: 0.75 }}>
        <Typography
          component="span"
          sx={{
            fontSize: '0.75rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.04em',
            color: 'text.secondary',
          }}
        >
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
          <InfoOutlinedIcon
            sx={{ fontSize: '1rem', color: 'text.secondary', cursor: 'help' }}
          />
        </Tooltip>
      </Stack>

      <Stack direction="row" spacing={1}>
        {PLANNER_OPTIONS.map((opt) => (
          <Button
            key={opt.value}
            variant={plannerMode === opt.value ? 'contained' : 'outlined'}
            size="small"
            disabled={disabled}
            onClick={() => onChange(opt.value)}
            sx={{ flex: 1, textTransform: 'none', flexDirection: 'column', py: 1 }}
          >
            <Typography component="span" sx={{ fontSize: '0.875rem', fontWeight: 600, lineHeight: 1.2 }}>
              {opt.label}
            </Typography>
            <Typography
              component="span"
              sx={{ fontSize: '0.6875rem', color: plannerMode === opt.value ? 'inherit' : 'text.secondary', lineHeight: 1.2, mt: '2px' }}
            >
              {opt.shortHint}
            </Typography>
          </Button>
        ))}
      </Stack>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Git tab content
// ---------------------------------------------------------------------------

interface GitTabProps {
  repoUrl: string;
  setRepoUrl: (v: string) => void;
  branch: string;
  setBranch: (v: string) => void;
  patSource: PatSource;
  setPatSource: (v: PatSource) => void;
  selectedPatId: string;
  setSelectedPatId: (v: string) => void;
  pastedPat: string;
  setPastedPat: (v: string) => void;
  urlError: string | null;
  disabled: boolean;
}

function GitTab({
  repoUrl, setRepoUrl,
  branch, setBranch,
  patSource, setPatSource,
  selectedPatId, setSelectedPatId,
  pastedPat, setPastedPat,
  urlError,
  disabled,
}: GitTabProps) {
  const { connections } = useConnections();
  const gitConnections = connections.filter((c) => c.provider === 'git');

  return (
    <Box>
      <TextField
        label="Repository URL"
        placeholder="https://github.com/owner/repo"
        value={repoUrl}
        onChange={(e) => setRepoUrl(e.target.value)}
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
        value={branch}
        onChange={(e) => setBranch(e.target.value)}
        fullWidth
        margin="normal"
        disabled={disabled}
        inputProps={{ 'data-testid': 'git-branch' }}
      />

      <FormControl fullWidth margin="normal" disabled={disabled}>
        <InputLabel id="pat-source-label">Authentication</InputLabel>
        <Select
          labelId="pat-source-label"
          value={patSource}
          label="Authentication"
          onChange={(e) => setPatSource(e.target.value as PatSource)}
          inputProps={{ 'data-testid': 'pat-source-select' }}
        >
          <MenuItem value="none">No auth (public repo)</MenuItem>
          <MenuItem value="stored">Use stored PAT</MenuItem>
          <MenuItem value="paste">Paste token once (not stored)</MenuItem>
        </Select>
      </FormControl>

      {patSource === 'stored' && gitConnections.length === 0 && (
        <Alert severity="info" sx={{ mt: 1 }}>
          No stored Git PATs found.{' '}
          <Link to="/settings?tab=connections" style={{ color: 'inherit' }}>
            Add a Git PAT in Settings
          </Link>
          .
        </Alert>
      )}

      {patSource === 'stored' && gitConnections.length > 0 && (
        <FormControl fullWidth margin="normal" disabled={disabled}>
          <InputLabel id="pat-select-label">Stored PAT</InputLabel>
          <Select
            labelId="pat-select-label"
            value={selectedPatId}
            label="Stored PAT"
            onChange={(e) => setSelectedPatId(e.target.value)}
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

      {patSource === 'paste' && (
        <TextField
          label="Personal Access Token"
          value={pastedPat}
          onChange={(e) => setPastedPat(e.target.value)}
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

// ---------------------------------------------------------------------------
// Confluence tab content
// ---------------------------------------------------------------------------

interface ConfluenceTabProps {
  spaceKeys: string[];
  setSpaceKeys: (v: string[]) => void;
  spaceKeysError: string | null;
  disabled: boolean;
}

function ConfluenceTab({ spaceKeys, setSpaceKeys, spaceKeysError, disabled }: ConfluenceTabProps) {
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
      <Box sx={{ mt: 2, mb: 1 }}>
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
        value={spaceKeys}
        disabled={disabled}
        onChange={(_e, value) => setSpaceKeys(value as string[])}
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
            placeholder='e.g. ENG, DOCS'
            margin="normal"
            error={!!spaceKeysError}
            helperText={spaceKeysError ?? 'Type a key and press Enter, or paste comma-separated keys'}
            inputProps={{ ...params.inputProps, 'data-testid': 'space-keys-input' }}
          />
        )}
        onInputChange={(_e, value, reason) => {
          // Handle comma-separated paste
          if (reason === 'input' && value.includes(',')) {
            const newKeys = value.split(',').map((k) => k.trim()).filter(Boolean);
            const merged = [...new Set([...spaceKeys, ...newKeys])];
            setSpaceKeys(merged);
          }
        }}
      />
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Jira tab content
// ---------------------------------------------------------------------------

interface JiraTabProps {
  jql: string;
  setJql: (v: string) => void;
  jqlError: string | null;
  disabled: boolean;
}

function JiraTab({ jql, setJql, jqlError, disabled }: JiraTabProps) {
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
      <Box sx={{ mt: 2, mb: 1 }}>
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
        value={jql}
        onChange={(e) => setJql(e.target.value)}
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

// ---------------------------------------------------------------------------
// Multi-source form
// ---------------------------------------------------------------------------

function MultiSourceGenerateForm({
  onSubmitMultiSource,
  disabled = false,
  initialUrl = '',
}: GenerateFormMultiSourceProps) {
  const [activeTab, setActiveTab] = useState<WikiSourceType>('git');
  const [plannerMode, setPlannerMode] = useState<PlannerMode>('agentic');
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Git state
  const [repoUrl, setRepoUrl] = useState(initialUrl);
  const [branch, setBranch] = useState('main');
  const [patSource, setPatSource] = useState<PatSource>('none');
  const [selectedPatId, setSelectedPatId] = useState('');
  const [pastedPat, setPastedPat] = useState('');

  // Confluence state
  const [spaceKeys, setSpaceKeys] = useState<string[]>([]);

  // Jira state
  const [jql, setJql] = useState('ORDER BY created DESC');

  // Validation
  const [touched, setTouched] = useState(false);

  const { atlassian, connections, refreshAtlassianIfNeeded } = useConnections();
  const gitConnections = connections.filter((c) => c.provider === 'git');

  // ---------------------------------------------------------------------------
  // Derived validation
  // ---------------------------------------------------------------------------

  const urlError = useMemo(() => {
    if (!touched) return null;
    if (activeTab !== 'git') return null;
    if (!repoUrl) return 'Repository URL is required';
    if (!isUrlish(repoUrl)) return 'Must be a valid URL (https://, git@, file://, or local path)';
    return null;
  }, [touched, activeTab, repoUrl]);

  const spaceKeysError = useMemo(() => {
    if (!touched) return null;
    if (activeTab !== 'confluence') return null;
    if (spaceKeys.length === 0) return 'At least one space key is required';
    return null;
  }, [touched, activeTab, spaceKeys]);

  const jqlError = useMemo(() => {
    if (!touched) return null;
    if (activeTab !== 'jira') return null;
    if (!jql.trim()) return 'JQL query is required';
    return null;
  }, [touched, activeTab, jql]);

  const atlassianRequired = activeTab === 'confluence' || activeTab === 'jira';
  const atlassianMissing = atlassianRequired && !atlassian;

  const isValid = useMemo(() => {
    if (atlassianMissing) return false;
    if (activeTab === 'git') {
      if (!repoUrl || !isUrlish(repoUrl)) return false;
      if (patSource === 'stored') {
        if (gitConnections.length === 0) return false;
        if (!selectedPatId) return false;
      }
      return true;
    }
    if (activeTab === 'confluence') return spaceKeys.length > 0;
    if (activeTab === 'jira') return jql.trim().length > 0;
    return false;
  }, [
    atlassianMissing, activeTab, repoUrl, patSource,
    gitConnections.length, selectedPatId, spaceKeys, jql,
  ]);

  // ---------------------------------------------------------------------------
  // Submit
  // ---------------------------------------------------------------------------

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setTouched(true);
      setSubmitError(null);

      if (!isValid) return;

      let body: GenerateWikiMultiSourceRequest;

      if (activeTab === 'git') {
        let pat: string | null = null;
        if (patSource === 'stored' && selectedPatId) {
          const conn = gitConnections.find((c) => c.id === selectedPatId);
          pat = conn?.pat ?? null;
        } else if (patSource === 'paste') {
          pat = pastedPat || null;
        }
        body = {
          source_type: 'git',
          scope: { repo_url: repoUrl, branch },
          auth: { pat },
          structure_planner: plannerMode,
        };
      } else {
        // Confluence or Jira — refresh Atlassian token first
        const freshAtlassian = await refreshAtlassianIfNeeded();
        if (!freshAtlassian) {
          setSubmitError('Atlassian connection lost. Please reconnect in Settings.');
          return;
        }

        const atlassianAuth: AtlassianAuth = {
          access_token: freshAtlassian.access_token,
          refresh_token: freshAtlassian.refresh_token,
          client_id: null,
        };

        const baseUrl = freshAtlassian.accessible_resources[0]?.url ?? freshAtlassian.site_name;

        if (activeTab === 'confluence') {
          body = {
            source_type: 'confluence',
            scope: { base_url: baseUrl, space_keys: spaceKeys },
            auth: atlassianAuth,
            structure_planner: plannerMode,
          };
        } else {
          body = {
            source_type: 'jira',
            scope: { base_url: baseUrl, jql },
            auth: atlassianAuth,
            structure_planner: plannerMode,
          };
        }
      }

      await onSubmitMultiSource(body);
    },
    [
      isValid, activeTab, patSource, selectedPatId, pastedPat,
      repoUrl, branch, gitConnections, plannerMode,
      refreshAtlassianIfNeeded, spaceKeys, jql, onSubmitMultiSource,
    ],
  );

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ maxWidth: '37.5rem', mx: 'auto' }}>
      <Tabs
        value={activeTab}
        onChange={(_e, val) => {
          setActiveTab(val as WikiSourceType);
          setTouched(false);
          setSubmitError(null);
        }}
        sx={{ mb: 1, borderBottom: 1, borderColor: 'divider' }}
        aria-label="source type"
      >
        <Tab label="Git" value="git" data-testid="tab-git" />
        <Tab label="Confluence" value="confluence" data-testid="tab-confluence" />
        <Tab label="Jira" value="jira" data-testid="tab-jira" />
      </Tabs>

      {activeTab === 'git' && (
        <GitTab
          repoUrl={repoUrl}
          setRepoUrl={setRepoUrl}
          branch={branch}
          setBranch={setBranch}
          patSource={patSource}
          setPatSource={setPatSource}
          selectedPatId={selectedPatId}
          setSelectedPatId={setSelectedPatId}
          pastedPat={pastedPat}
          setPastedPat={setPastedPat}
          urlError={urlError}
          disabled={disabled}
        />
      )}

      {activeTab === 'confluence' && (
        <ConfluenceTab
          spaceKeys={spaceKeys}
          setSpaceKeys={setSpaceKeys}
          spaceKeysError={spaceKeysError}
          disabled={disabled}
        />
      )}

      {activeTab === 'jira' && (
        <JiraTab
          jql={jql}
          setJql={setJql}
          jqlError={jqlError}
          disabled={disabled}
        />
      )}

      <PlannerSection
        plannerMode={plannerMode}
        onChange={setPlannerMode}
        disabled={disabled}
      />

      {submitError && (
        <Alert severity="error" sx={{ mt: 2 }} data-testid="submit-error">
          {submitError}
        </Alert>
      )}

      <Button
        type="submit"
        variant="contained"
        size="large"
        fullWidth
        sx={{ mt: 2.5 }}
        disabled={disabled || atlassianMissing}
        data-testid="generate-submit"
      >
        Generate Wiki
      </Button>

      {atlassianMissing && (
        <FormHelperText error sx={{ textAlign: 'center', mt: 0.5 }}>
          Connect to Atlassian in{' '}
          <Link to="/settings?tab=connections" style={{ color: 'inherit' }}>
            Settings
          </Link>{' '}
          to continue.
        </FormHelperText>
      )}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Public export — discriminated union dispatch
// ---------------------------------------------------------------------------

export function GenerateForm(props: GenerateFormProps) {
  if (props.onSubmitMultiSource) {
    return (
      <MultiSourceGenerateForm
        onSubmitMultiSource={props.onSubmitMultiSource}
        disabled={props.disabled}
        initialUrl={props.initialUrl}
      />
    );
  }
  return (
    <LegacyGenerateForm
      onSubmit={props.onSubmit}
      disabled={props.disabled}
      initialUrl={props.initialUrl}
    />
  );
}
