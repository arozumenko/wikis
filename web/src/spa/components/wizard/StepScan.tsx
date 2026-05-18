/**
 * Step 3 of the AddSource wizard (#208): preview the source.
 *
 * Calls ``POST /api/v1/sources/scan`` and renders the returned preview.
 * For Confluence / Jira the backend currently returns 501 (#211 ships
 * those) — surface that as a friendly "preview not supported yet" panel
 * with a Skip-to-Confirm affordance. The Skip control is always available
 * so a user who already trusts their inputs can bypass the scan even on
 * git.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Divider,
  Stack,
  Typography,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import FolderOutlinedIcon from '@mui/icons-material/FolderOutlined';
import { ApiError } from '../../api/client';
import { hashScanRequest, scanSource } from '../../api/wiki';
import type {
  ConfluenceScanPreview,
  GitScanPreview,
  JiraScanPreview,
  ScanRequest,
  ScanResponse,
} from '../../api/wiki';

interface StepScanProps {
  buildScanRequest: () => ScanRequest | null;
  onScanComplete: (result: ScanResponse | null) => void;
  /**
   * If the container already holds a scan result whose scope matches the
   * current ``buildScanRequest()`` payload, the step renders it directly
   * without re-hitting the network. Prevents Back→Next from re-scanning
   * a remote repo on every visit (Rio review on #216).
   */
  cachedResult?: ScanResponse | null;
  cachedScopeHash?: string | null;
}

type ScanState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'success'; result: ScanResponse }
  | { kind: 'unsupported'; message: string }
  | { kind: 'error'; message: string };

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function hashRequest(req: ScanRequest | null): string | null {
  if (!req) return null;
  // Delegate to the shared helper (#217) so auth presence is factored in.
  return hashScanRequest(req);
}

export function StepScan({
  buildScanRequest,
  onScanComplete,
  cachedResult,
  cachedScopeHash,
}: StepScanProps) {
  const [state, setState] = useState<ScanState>({ kind: 'idle' });
  // C4 / Rio review: guard the post-await setState + onScanComplete calls
  // so they no-op after unmount (user clicked Back / Skip mid-flight).
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const runScan = useCallback(async () => {
    const req = buildScanRequest();
    if (!req) {
      if (mountedRef.current)
        setState({ kind: 'error', message: 'Configuration is invalid — go back and fix it.' });
      // C3: clear stale container result so Confirm can't show a previous
      // preview after the new config has invalidated it.
      onScanComplete(null);
      return;
    }
    if (mountedRef.current) setState({ kind: 'loading' });
    onScanComplete(null); // C3: also clear at start of any new scan attempt.
    try {
      const result = await scanSource(req);
      if (!mountedRef.current) return;
      setState({ kind: 'success', result });
      onScanComplete(result);
    } catch (err) {
      if (!mountedRef.current) return;
      if (err instanceof ApiError) {
        // 501 — unimplemented for this source type (Confluence / Jira until #211).
        if (err.status === 501) {
          const message =
            typeof err.body === 'object' && err.body !== null && 'detail' in err.body
              ? String((err.body as { detail: unknown }).detail)
              : 'Preview not supported for this source yet.';
          setState({ kind: 'unsupported', message });
          // Treat as "no preview available" — Skip-to-Confirm still works.
          onScanComplete(null);
          return;
        }
        // 400 — scan returned an actionable error (auth, unreachable, etc).
        const detail = (err.body as { detail?: { error?: string; reachable?: boolean } } | null)
          ?.detail;
        const message =
          (typeof detail === 'object' && detail !== null && detail.error) ||
          err.message ||
          'Scan failed.';
        setState({ kind: 'error', message });
        return;
      }
      setState({
        kind: 'error',
        message: err instanceof Error ? err.message : 'Scan failed.',
      });
    }
  }, [buildScanRequest, onScanComplete]);

  // Auto-run on mount unless the container already has a fresh result for
  // the same scope (Rio: Back→Next must not re-hit the network).
  // ``runScan`` is intentionally NOT in the dep array — re-running on
  // every re-render would hammer the backend.
  useEffect(() => {
    const currentHash = hashRequest(buildScanRequest());
    if (
      currentHash &&
      cachedScopeHash &&
      cachedScopeHash === currentHash &&
      cachedResult !== undefined
    ) {
      // Reuse the cached result. State must reflect it so Retry still works.
      if (cachedResult) {
        setState({ kind: 'success', result: cachedResult });
      } else {
        // Cached "no preview" (e.g. earlier 501) — render the unsupported
        // affordance again so the user has a Skip option.
        setState({ kind: 'unsupported', message: 'Preview not available for this source.' });
      }
      return;
    }
    void runScan();
  }, []);

  if (state.kind === 'loading') {
    return (
      <Stack alignItems="center" spacing={2} sx={{ py: 6 }}>
        <CircularProgress />
        <Typography variant="body2" color="text.secondary">
          Validating source and previewing content…
        </Typography>
      </Stack>
    );
  }

  if (state.kind === 'unsupported') {
    return (
      <Alert severity="info" sx={{ mt: 1 }} data-testid="scan-unsupported">
        {state.message}
        <br />
        You can still continue to the next step and submit without a preview.
      </Alert>
    );
  }

  if (state.kind === 'error') {
    return (
      <Stack spacing={2} sx={{ mt: 1 }}>
        <Alert severity="error" data-testid="scan-error">
          {state.message}
        </Alert>
        <Button
          variant="outlined"
          onClick={() => void runScan()}
          sx={{ alignSelf: 'flex-start' }}
        >
          Retry scan
        </Button>
      </Stack>
    );
  }

  if (state.kind === 'success') {
    const p = state.result.preview;
    if (!p) {
      return (
        <Alert severity="info" data-testid="scan-no-preview">
          Source is reachable but no preview was returned.
        </Alert>
      );
    }

    const header = (
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
        <CheckCircleIcon color="success" />
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
          Source reachable
        </Typography>
      </Stack>
    );

    const warnings =
      state.result.warnings.length > 0 ? (
        <Alert severity="warning" sx={{ mt: 2 }}>
          {state.result.warnings.map((w) => (
            <Box key={w}>{w}</Box>
          ))}
        </Alert>
      ) : null;

    // Git preview (#221)
    if (isGitPreview(state.result)) {
      const gp = state.result.preview as GitScanPreview;
      return (
        <Box sx={{ mt: 1 }} data-testid="scan-success">
          {header}
          <Stack direction="row" spacing={3} sx={{ mb: 2 }}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Branch
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {gp.resolved_branch}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Files
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {gp.file_count.toLocaleString()}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Size
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {formatBytes(gp.size_bytes)}
              </Typography>
            </Box>
            {gp.commit_hash && (
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Commit
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 500 }}>
                  {gp.commit_hash.slice(0, 7)}
                </Typography>
              </Box>
            )}
          </Stack>

          {gp.top_paths.length > 0 && (
            <>
              <Divider sx={{ my: 1.5 }} />
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ display: 'block', mb: 1 }}
              >
                Top-level entries
              </Typography>
              <Stack direction="row" spacing={0.75} sx={{ flexWrap: 'wrap', gap: 0.75 }}>
                {gp.top_paths.map((path) => (
                  <Chip
                    key={path}
                    icon={path.endsWith('/') ? <FolderOutlinedIcon /> : undefined}
                    label={path}
                    size="small"
                    variant="outlined"
                  />
                ))}
              </Stack>
            </>
          )}
          {warnings}
        </Box>
      );
    }

    // Confluence preview (#221)
    if (isConfluencePreview(state.result)) {
      const cp = state.result.preview as ConfluenceScanPreview;
      return (
        <Box sx={{ mt: 1 }} data-testid="scan-success">
          {header}
          <Stack direction="row" spacing={3} sx={{ mb: 2 }}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Total pages
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {cp.total_pages != null ? cp.total_pages.toLocaleString() : '?'}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Spaces
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {cp.spaces.length}
              </Typography>
            </Box>
          </Stack>

          {cp.spaces.length > 0 && (
            <>
              <Divider sx={{ my: 1.5 }} />
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ display: 'block', mb: 1 }}
              >
                Spaces
              </Typography>
              <Stack direction="row" spacing={0.75} sx={{ flexWrap: 'wrap', gap: 0.75 }}>
                {cp.spaces.map((space) => (
                  <Chip
                    key={space.key}
                    label={
                      space.page_count != null
                        ? `${space.name} (${space.page_count.toLocaleString()} pages)`
                        : `${space.name} (?)`
                    }
                    size="small"
                    variant="outlined"
                    data-testid={`confluence-space-${space.key}`}
                  />
                ))}
              </Stack>
            </>
          )}
          {warnings}
        </Box>
      );
    }

    // Jira preview (#221)
    if (isJiraPreview(state.result)) {
      const jp = state.result.preview as JiraScanPreview;
      return (
        <Box sx={{ mt: 1 }} data-testid="scan-success">
          {header}
          <Stack direction="row" spacing={3} sx={{ mb: 2 }}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Matching issues
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {jp.matching_issues.toLocaleString()}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                JQL validated
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {jp.jql_validated ? '✓ Yes' : '✗ No'}
              </Typography>
            </Box>
          </Stack>

          {jp.sample_issue_keys.length > 0 && (
            <>
              <Divider sx={{ my: 1.5 }} />
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ display: 'block', mb: 1 }}
              >
                Sample issues
              </Typography>
              <Stack direction="row" spacing={0.75} sx={{ flexWrap: 'wrap', gap: 0.75 }}>
                {jp.sample_issue_keys.map((key) => (
                  <Chip
                    key={key}
                    label={key}
                    size="small"
                    variant="outlined"
                    data-testid={`jira-issue-${key}`}
                  />
                ))}
              </Stack>
            </>
          )}
          {warnings}
        </Box>
      );
    }

    // Fallback — unknown preview shape (defensive)
    return (
      <Alert severity="info" data-testid="scan-no-preview">
        Source is reachable but preview type is unrecognised.
      </Alert>
    );
  }

  return null;
}

// ---------------------------------------------------------------------------
// Type guards for preview discriminated union (#221)
// ---------------------------------------------------------------------------

function isGitPreview(
  result: ScanResponse,
): result is ScanResponse & { preview: GitScanPreview } {
  const p = result.preview;
  return (
    p !== null &&
    result.source_type === 'git' &&
    'resolved_branch' in p
  );
}

function isConfluencePreview(
  result: ScanResponse,
): result is ScanResponse & { preview: ConfluenceScanPreview } {
  const p = result.preview;
  return (
    p !== null &&
    result.source_type === 'confluence' &&
    'spaces' in p
  );
}

function isJiraPreview(
  result: ScanResponse,
): result is ScanResponse & { preview: JiraScanPreview } {
  const p = result.preview;
  return (
    p !== null &&
    result.source_type === 'jira' &&
    'matching_issues' in p
  );
}
