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

import { useCallback, useEffect, useState } from 'react';
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
import { scanSource } from '../../api/wiki';
import type { ScanRequest, ScanResponse } from '../../api/wiki';

interface StepScanProps {
  buildScanRequest: () => ScanRequest | null;
  onScanComplete: (result: ScanResponse | null) => void;
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

export function StepScan({ buildScanRequest, onScanComplete }: StepScanProps) {
  const [state, setState] = useState<ScanState>({ kind: 'idle' });

  const runScan = useCallback(async () => {
    const req = buildScanRequest();
    if (!req) {
      setState({ kind: 'error', message: 'Configuration is invalid — go back and fix it.' });
      return;
    }
    setState({ kind: 'loading' });
    try {
      const result = await scanSource(req);
      setState({ kind: 'success', result });
      onScanComplete(result);
    } catch (err) {
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

  // Auto-run on mount so the user sees a result without an extra click.
  // Re-run is via the "Retry" button on error states.
  //
  // ``runScan`` is intentionally NOT in the dep array — toggling between
  // steps would otherwise re-fire and hammer the backend.
  useEffect(() => {
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
    return (
      <Box sx={{ mt: 1 }} data-testid="scan-success">
        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
          <CheckCircleIcon color="success" />
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            Source reachable
          </Typography>
        </Stack>

        <Stack direction="row" spacing={3} sx={{ mb: 2 }}>
          <Box>
            <Typography variant="caption" color="text.secondary">
              Branch
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>
              {p.resolved_branch}
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">
              Files
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>
              {p.file_count.toLocaleString()}
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">
              Size
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>
              {formatBytes(p.size_bytes)}
            </Typography>
          </Box>
          {p.commit_hash && (
            <Box>
              <Typography variant="caption" color="text.secondary">
                Commit
              </Typography>
              <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 500 }}>
                {p.commit_hash.slice(0, 7)}
              </Typography>
            </Box>
          )}
        </Stack>

        {p.top_paths.length > 0 && (
          <>
            <Divider sx={{ my: 1.5 }} />
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              Top-level entries
            </Typography>
            <Stack direction="row" spacing={0.75} sx={{ flexWrap: 'wrap', gap: 0.75 }}>
              {p.top_paths.map((path) => (
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

        {state.result.warnings.length > 0 && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            {state.result.warnings.map((w) => (
              <Box key={w}>{w}</Box>
            ))}
          </Alert>
        )}
      </Box>
    );
  }

  return null;
}
