import { useCallback, useMemo, useState } from 'react';
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
  Tab,
  Tabs,
  TextField,
  Typography,
} from '@mui/material';
import { useConnections } from '../hooks/useConnections';
import type { AtlassianAuth, GenerateWikiMultiSourceRequest, WikiSourceType } from '../api/wiki';
import type { components } from '../api/types.generated';

// ---------------------------------------------------------------------------
// Legacy shape — kept for backward compat with GeneratePage (direct URL mode)
// ---------------------------------------------------------------------------
type GenerateWikiRequest = components['schemas']['GenerateWikiRequest'];

// ---------------------------------------------------------------------------
// PAT source options (Git tab)
// ---------------------------------------------------------------------------

type PatSource = 'paste' | 'none';

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

  const provider = detectProvider(repoUrl);
  const isLocal = provider === 'local';

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      // The unified pipeline is the only planner path (#242).  The
      // backend still accepts ``structure_planner`` / ``planner_type``
      // for one release cycle (logged as deprecated) — the UI no
      // longer offers a choice but the generated OpenAPI types still
      // mark them required-with-null, so we pass null explicitly.
      onSubmit({
        source_type: 'git',
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
        structure_planner: null,
        planner_type: null,
        exclude_tests: null,
      });
    },
    [repoUrl, branch, provider, isLocal, accessToken, onSubmit],
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
// Git tab content
// ---------------------------------------------------------------------------

interface GitTabProps {
  repoUrl: string;
  setRepoUrl: (v: string) => void;
  branch: string;
  setBranch: (v: string) => void;
  patSource: PatSource;
  setPatSource: (v: PatSource) => void;
  pastedPat: string;
  setPastedPat: (v: string) => void;
  urlError: string | null;
  disabled: boolean;
}

function GitTab({
  repoUrl, setRepoUrl,
  branch, setBranch,
  patSource, setPatSource,
  pastedPat, setPastedPat,
  urlError,
  disabled,
}: GitTabProps) {
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
          <MenuItem value="paste">Paste token once (not stored)</MenuItem>
        </Select>
      </FormControl>

      {patSource === 'paste' && (
        <TextField
          label="Personal Access Token"
          value={pastedPat}
          onChange={(e) => setPastedPat(e.target.value)}
          required
          fullWidth
          margin="normal"
          type="password"
          disabled={disabled}
          error={!pastedPat.trim()}
          helperText={
            !pastedPat.trim()
              ? 'Token is required for the "Paste token once" option'
              : 'Token will not be stored'
          }
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
          No Atlassian connection found. Use the &ldquo;Add a source&rdquo; wizard to
          connect to Atlassian.
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
          No Atlassian connection found. Use the &ldquo;Add a source&rdquo; wizard to
          connect to Atlassian.
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
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Git state
  const [repoUrl, setRepoUrl] = useState(initialUrl);
  const [branch, setBranch] = useState('main');
  const [patSource, setPatSource] = useState<PatSource>('none');
  const [pastedPat, setPastedPat] = useState('');

  // Confluence state
  const [spaceKeys, setSpaceKeys] = useState<string[]>([]);

  // Jira state
  const [jql, setJql] = useState('ORDER BY created DESC');

  // Validation
  const [touched, setTouched] = useState(false);

  const { atlassian, refreshAtlassianIfNeeded } = useConnections();

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
      // "Paste token once" must require an actual token — empty would
      // silently submit as no-auth (Copilot on PR #276).
      if (patSource === 'paste' && !pastedPat.trim()) return false;
      return true;
    }
    if (activeTab === 'confluence') return spaceKeys.length > 0;
    if (activeTab === 'jira') return jql.trim().length > 0;
    return false;
  }, [atlassianMissing, activeTab, repoUrl, patSource, pastedPat, spaceKeys, jql]);

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
        const pat: string | null = patSource === 'paste' ? pastedPat || null : null;
        body = {
          source_type: 'git',
          scope: { repo_url: repoUrl, branch },
          auth: { pat },
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
          };
        } else {
          body = {
            source_type: 'jira',
            scope: { base_url: baseUrl, jql },
            auth: atlassianAuth,
          };
        }
      }

      await onSubmitMultiSource(body);
    },
    [
      isValid, activeTab, patSource, pastedPat,
      repoUrl, branch,
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
        disabled={disabled || atlassianMissing || !isValid}
        data-testid="generate-submit"
      >
        Generate Wiki
      </Button>

      {atlassianMissing && (
        <FormHelperText error sx={{ textAlign: 'center', mt: 0.5 }}>
          Connect to Atlassian via the &ldquo;Add a source&rdquo; wizard to continue.
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
