import { Box, Chip, Tooltip, Typography } from '@mui/material';
import DescriptionIcon from '@mui/icons-material/Description';
import VerifiedIcon from '@mui/icons-material/Verified';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

export interface SourceRef {
  file_path: string;
  line_start?: number | null;
  line_end?: number | null;
  /**
   * Edge-confidence label propagated from the underlying graph
   * (#120). `EXTRACTED` = direct parser observation, `INFERRED` =
   * name-only resolution, `AMBIGUOUS` = reserved. `null` / missing
   * means the citation came from a path with no edge metadata
   * (e.g. a pure FTS hit) — render without an indicator.
   */
  confidence?: string | null;
}

interface CitationChipsProps {
  sources: SourceRef[];
  repoUrl?: string;
  branch?: string;
}

function isLocalRepo(repoUrl?: string): boolean {
  return !!repoUrl && (repoUrl.startsWith('/') || repoUrl.startsWith('file://'));
}

// Only emit chip links for URLs we can prove are safe to put in an
// `<a href>`. `repo_url` is server-supplied opaque text, so without
// this guard a malicious / mis-stored value like `javascript:…` would
// execute on click. The backend validates on ingestion but we don't
// rely on that here.
function isSafeHttpUrl(value: string): boolean {
  try {
    const { protocol } = new URL(value);
    return protocol === 'https:' || protocol === 'http:';
  } catch {
    return false;
  }
}

function buildUrl(
  repoUrl: string,
  filePath: string,
  branch: string,
  lineStart?: number | null,
): string {
  const line = lineStart ? `#L${lineStart}` : '';
  return `${repoUrl}/blob/${branch}/${encodeURI(filePath)}${line}`;
}

function shortPath(filePath: string): string {
  const parts = filePath.split('/');
  if (parts.length <= 2) return filePath;
  return `.../${parts.slice(-2).join('/')}`;
}

// #120 Phase 3: per-confidence rendering metadata. Keeping the
// definitions next to the component (rather than a separate const
// module) since they're tightly coupled to the chip's visual style.
interface ConfidenceMeta {
  /** MUI chip color token. */
  color: 'success' | 'warning' | 'default';
  /** Single short word for the label suffix. */
  label: string;
  /** Hover tooltip body. */
  tooltip: string;
  /** Optional leading icon override (otherwise we use DescriptionIcon). */
  icon: React.ReactElement;
}

const CONFIDENCE_META: Record<string, ConfidenceMeta> = {
  EXTRACTED: {
    color: 'success',
    label: 'verified',
    tooltip:
      'Verified: the underlying graph edge was directly observed by the parser. Highest trust signal.',
    icon: <VerifiedIcon sx={{ fontSize: 14 }} />,
  },
  INFERRED: {
    color: 'warning',
    label: 'inferred',
    tooltip:
      'Inferred: the edge was resolved by name only (no type context or cross-file disambiguation). Useful but may cite the wrong target when symbol names collide.',
    icon: <HelpOutlineIcon sx={{ fontSize: 14 }} />,
  },
  AMBIGUOUS: {
    color: 'default',
    label: 'ambiguous',
    tooltip:
      'Ambiguous: multiple plausible targets matched. Reserved for future use; treat similar to inferred for now.',
    icon: <HelpOutlineIcon sx={{ fontSize: 14 }} />,
  },
};

function metaFor(confidence?: string | null): ConfidenceMeta | null {
  if (!confidence) return null;
  // Backend canonicalises to uppercase; tolerate lowercase too.
  return CONFIDENCE_META[confidence.toUpperCase()] ?? null;
}

export function CitationChips({ sources, repoUrl, branch = 'main' }: CitationChipsProps) {
  if (sources.length === 0) return null;

  return (
    <Box sx={{ mt: 4, pt: 3, borderTop: '1px solid', borderColor: 'divider' }}>
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ mb: 1, display: 'block', fontWeight: 500 }}
      >
        Sources ({sources.length})
      </Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
        {sources.map((src, i) => {
          const label = `${shortPath(src.file_path)}${src.line_start ? `:${src.line_start}` : ''}`;
          const url =
            repoUrl && !isLocalRepo(repoUrl) && isSafeHttpUrl(repoUrl)
              ? buildUrl(repoUrl, src.file_path, branch, src.line_start)
              : undefined;

          const meta = metaFor(src.confidence);
          // Stable key: file_path + line is unique enough for the
          // chip list. Tie-break with `i` for duplicate citations.
          const key = `${src.file_path}:${src.line_start ?? ''}:${i}`;

          const chip = (
            <Chip
              icon={meta?.icon ?? <DescriptionIcon sx={{ fontSize: 14 }} />}
              label={label}
              size="small"
              variant="outlined"
              color={meta?.color ?? 'default'}
              component={url ? 'a' : 'span'}
              href={url}
              target="_blank"
              // `noreferrer` strips Referer so we don't leak internal
              // URLs to GitHub/GitLab/etc; `noopener` prevents
              // window.opener access.
              rel="noopener noreferrer"
              clickable={!!url}
              sx={{
                fontFamily: 'monospace',
                fontSize: '0.72rem',
                height: 26,
                '& .MuiChip-icon': { ml: 0.5 },
              }}
            />
          );

          // Wrap in a Tooltip only when we have a confidence label
          // to explain — keeps the bare chip behavior unchanged for
          // citations without graph-edge confidence.
          return meta ? (
            <Tooltip key={key} title={meta.tooltip} arrow placement="top">
              {/* Tooltip needs a real DOM child wrapper for the
                  anchor element when the inner element doesn't
                  forward refs (the Chip-as-link case). */}
              <span style={{ display: 'inline-flex' }}>{chip}</span>
            </Tooltip>
          ) : (
            <Box key={key} component="span" sx={{ display: 'inline-flex' }}>
              {chip}
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}
