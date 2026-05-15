/**
 * Banner that surfaces the result of an incremental wiki refresh (#116 PR 5).
 *
 * Subscribes to the invocation's SSE stream and renders:
 *   - live per-page status while the run is in flight, and
 *   - a "N unchanged · M patched · K regenerated" summary once
 *     the `incremental_summary` event arrives.
 *
 * Designed to render inline (e.g. below the wiki page list) and quietly
 * disappear when the run completes — the user gets feedback without a
 * modal.
 */

import { useEffect, useMemo, useState } from 'react';
import { Alert, Box, Chip, LinearProgress, Typography } from '@mui/material';

import { subscribeSSE, type IncrementalRegenStats, type SSEEventData } from '../api/sse';

interface IncrementalRefreshBannerProps {
  invocationId: string;
  /** How many pages the backend reported it would consider, for the
   *  in-flight progress bar denominator. */
  totalPages: number;
  /** Optional close callback for the dismiss button. */
  onDismiss?: () => void;
}

interface PageEvent {
  kind: 'unchanged' | 'patched' | 'edited' | 'regenerated';
  pageId: string;
  pageTitle: string;
}

export function IncrementalRefreshBanner({
  invocationId,
  totalPages,
  onDismiss,
}: IncrementalRefreshBannerProps): JSX.Element | null {
  const [events, setEvents] = useState<PageEvent[]>([]);
  const [stats, setStats] = useState<IncrementalRegenStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const sub = subscribeSSE(
      `/api/v1/invocations/${invocationId}/stream`,
      (event: SSEEventData) => {
        switch (event.type) {
          case 'page_unchanged':
            setEvents((prev) => [
              ...prev,
              {
                kind: 'unchanged',
                pageId: event._pageId ?? '',
                pageTitle: event._pageTitle ?? '',
              },
            ]);
            break;
          case 'page_patched':
            setEvents((prev) => [
              ...prev,
              {
                kind: 'patched',
                pageId: event._pageId ?? '',
                pageTitle: event._pageTitle ?? '',
              },
            ]);
            break;
          case 'page_edited':
            setEvents((prev) => [
              ...prev,
              {
                kind: 'edited',
                pageId: event._pageId ?? '',
                pageTitle: event._pageTitle ?? '',
              },
            ]);
            break;
          case 'page_regenerated':
            setEvents((prev) => [
              ...prev,
              {
                kind: 'regenerated',
                pageId: event._pageId ?? '',
                pageTitle: event._pageTitle ?? '',
              },
            ]);
            break;
          case 'incremental_summary':
            setStats(event.stats);
            break;
          case 'task_failed':
            setError(event.error);
            break;
        }
      },
      (err) => {
        setError(err instanceof Error ? err.message : 'SSE stream error');
      },
    );
    return () => sub.close();
  }, [invocationId]);

  const inFlight = stats === null && error === null;
  const progressFraction = useMemo(() => {
    if (totalPages === 0) return 0;
    return Math.min(events.length / totalPages, 1);
  }, [events.length, totalPages]);

  if (error) {
    return (
      <Alert severity="error" onClose={onDismiss} sx={{ mt: 2 }}>
        Incremental refresh failed: {error}
      </Alert>
    );
  }

  if (inFlight) {
    return (
      <Box
        sx={{
          mt: 2,
          p: 2,
          border: 1,
          borderColor: 'divider',
          borderRadius: 1,
        }}
      >
        <Typography variant="body2" gutterBottom>
          Incremental refresh — {events.length} / {totalPages} pages processed
        </Typography>
        <LinearProgress variant="determinate" value={progressFraction * 100} />
      </Box>
    );
  }

  // stats is non-null past this point.
  const savedFraction = computeSavedFraction(stats!);
  return (
    <Alert
      severity={stats!.structural_failed > 0 ? 'warning' : 'success'}
      onClose={onDismiss}
      sx={{ mt: 2 }}
    >
      <Typography variant="body2" sx={{ fontWeight: 500, mb: 0.5 }}>
        Incremental refresh complete
        {savedFraction > 0 ? ` — saved ~${Math.round(savedFraction * 100)}% LLM work` : ''}
      </Typography>
      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
        <StatChip label="unchanged" count={stats!.unchanged} color="default" />
        <StatChip label="patched" count={stats!.trivial_patched} color="info" />
        <StatChip label="edited" count={stats!.edit_applied} color="primary" />
        <StatChip
          label="regenerated"
          count={stats!.structural_regenerated}
          color="secondary"
        />
        {stats!.structural_failed > 0 ? (
          <StatChip label="failed" count={stats!.structural_failed} color="error" />
        ) : null}
      </Box>
    </Alert>
  );
}

function StatChip({
  label,
  count,
  color,
}: {
  label: string;
  count: number;
  color: 'default' | 'primary' | 'secondary' | 'info' | 'error';
}): JSX.Element | null {
  if (count === 0) return null;
  return (
    <Chip
      size="small"
      label={`${count} ${label}`}
      color={color}
      variant={color === 'default' ? 'outlined' : 'filled'}
    />
  );
}

/**
 * Rough "LLM work saved" fraction. Trivial = 0 cost; edited = ~25% cost;
 * structural = full cost. The exact savings depend on prompt sizes, but
 * this gives the user a directional signal.
 */
function computeSavedFraction(stats: IncrementalRegenStats): number {
  if (stats.total_pages === 0) return 0;
  const fullCost = stats.total_pages;
  const actualCost =
    stats.trivial_patched * 0 +
    stats.edit_applied * 0.25 +
    stats.structural_regenerated * 1 +
    stats.structural_failed * 1; // a failed structural still cost the attempt
  return Math.max(0, 1 - actualCost / fullCost);
}
