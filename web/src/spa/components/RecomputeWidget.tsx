import { useCallback, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  IconButton,
  LinearProgress,
  Paper,
  Stack,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import RefreshIcon from '@mui/icons-material/Refresh';

import { subscribeRecomputeSSE } from '../api/sse';
import type { RecomputeProgressEvent } from '../api/sse';

interface RecomputeWidgetProps {
  projectId: string;
  isOwner: boolean;
}

interface PhaseRow {
  phase: string;
  label: string;
  progress: number;
  total: number;
  message: string;
}

const PHASE_LABELS: Record<string, string> = {
  project_relatedness_progress: 'Repo relatedness',
  cross_repo_linker_progress: 'Cross-repo linker',
  project_clustering_progress: 'Project clustering',
};

/**
 * PR-16 — Project recompute progress widget.
 * Mirrors the wiki-generation progress UI: triggers POST
 * /api/v1/projects/{id}/recompute and renders per-phase progress
 * bars from the SSE stream.
 */
export function RecomputeWidget({ projectId, isOwner }: RecomputeWidgetProps) {
  const [running, setRunning] = useState(false);
  const [phases, setPhases] = useState<Record<string, PhaseRow>>({});
  const [summary, setSummary] = useState<RecomputeProgressEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  const handleEvent = useCallback((evt: RecomputeProgressEvent) => {
    if (evt.type === 'recompute_complete') {
      setSummary(evt);
      return;
    }
    if (evt.type === 'recompute_error') {
      setError(evt.error || 'Recompute failed');
      return;
    }
    const label = PHASE_LABELS[evt.type];
    if (!label) return;
    setPhases((prev) => ({
      ...prev,
      [evt.type]: {
        phase: evt.type,
        label,
        progress: evt.progress ?? 0,
        total: evt.total ?? 1,
        message: evt.message ?? '',
      },
    }));
  }, []);

  const handleStart = useCallback(() => {
    setRunning(true);
    setPhases({});
    setSummary(null);
    setError(null);
    cancelRef.current = subscribeRecomputeSSE(
      projectId,
      handleEvent,
      () => setRunning(false),
      (e) => {
        setError(e instanceof Error ? e.message : String(e));
        setRunning(false);
      },
    );
  }, [projectId, handleEvent]);

  const handleCancel = useCallback(() => {
    cancelRef.current?.();
    cancelRef.current = null;
    setRunning(false);
  }, []);

  const handleDismiss = useCallback(() => {
    setPhases({});
    setSummary(null);
    setError(null);
  }, []);

  const phaseRows = Object.values(phases);
  const hasContent = running || phaseRows.length > 0 || summary || error;

  if (!isOwner && !hasContent) return null;

  return (
    <Paper
      variant="outlined"
      sx={{
        mb: 3,
        p: 2,
        borderRadius: 2,
        borderColor: error ? 'error.main' : 'divider',
      }}
    >
      <Stack
        direction="row"
        alignItems="center"
        justifyContent="space-between"
        sx={{ mb: phaseRows.length > 0 || summary || error ? 1.5 : 0 }}
      >
        <Stack direction="row" alignItems="center" spacing={1}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            Cross-repo recompute
          </Typography>
          {running && <CircularProgress size={16} />}
        </Stack>
        <Stack direction="row" spacing={1}>
          {isOwner && !running && (
            <Button
              size="small"
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={handleStart}
            >
              Recompute now
            </Button>
          )}
          {running && (
            <Button size="small" variant="outlined" color="warning" onClick={handleCancel}>
              Cancel
            </Button>
          )}
          {hasContent && !running && (
            <IconButton size="small" onClick={handleDismiss} aria-label="dismiss recompute panel">
              <CloseIcon fontSize="small" />
            </IconButton>
          )}
        </Stack>
      </Stack>

      {phaseRows.map((row) => {
        const pct = row.total > 0 ? Math.min(100, Math.round((row.progress / row.total) * 100)) : 0;
        return (
          <Box key={row.phase} sx={{ mb: 1.25 }}>
            <Stack direction="row" justifyContent="space-between" sx={{ mb: 0.25 }}>
              <Typography variant="caption" sx={{ fontWeight: 600 }}>
                {row.label}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {row.progress}/{row.total}
              </Typography>
            </Stack>
            <LinearProgress variant="determinate" value={pct} />
            {row.message && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.25, display: 'block' }}>
                {row.message}
              </Typography>
            )}
          </Box>
        );
      })}

      {summary && (
        <Alert severity="success" sx={{ mt: 1 }}>
          Recompute complete — {summary.wiki_count ?? 0} wikis,{' '}
          {summary.pair_count ?? 0} pairs, {summary.edge_count ?? 0} cross-repo edges
          {summary.community_count ? `, ${summary.community_count} communities` : ''}.
        </Alert>
      )}
      {error && (
        <Alert severity="error" sx={{ mt: 1 }}>
          {error}
        </Alert>
      )}
    </Paper>
  );
}
