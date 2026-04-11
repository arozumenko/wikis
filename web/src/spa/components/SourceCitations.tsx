import { useState } from 'react';
import { Box, Chip, Collapse, Link, Typography } from '@mui/material';
import SourceIcon from '@mui/icons-material/Description';

interface SourceFile {
  file_path: string;
  line_start?: number | null;
  line_end?: number | null;
  /** Present on cross-repo (project-level) answers — identifies the source wiki. */
  wiki_id?: string | null;
  /** Human-readable wiki title for cross-repo attribution. */
  wiki_title?: string | null;
}

interface SourceCitationsProps {
  sources: SourceFile[];
  repoUrl?: string;
  branch?: string;
  provider?: string;
}

function buildSourceUrl(
  repoUrl: string,
  filePath: string,
  branch: string,
  provider: string,
  lineStart?: number | null,
  lineEnd?: number | null,
): string {
  const lineRange = lineStart
    ? provider === 'github'
      ? `#L${lineStart}${lineEnd ? `-L${lineEnd}` : ''}`
      : provider === 'gitlab'
        ? `#L${lineStart}${lineEnd ? `-${lineEnd}` : ''}`
        : provider === 'bitbucket'
          ? `#lines-${lineStart}${lineEnd ? `:${lineEnd}` : ''}`
          : `#L${lineStart}`
    : '';

  switch (provider) {
    case 'github':
      return `${repoUrl}/blob/${branch}/${filePath}${lineRange}`;
    case 'gitlab':
      return `${repoUrl}/-/blob/${branch}/${filePath}${lineRange}`;
    case 'bitbucket':
      return `${repoUrl}/src/${branch}/${filePath}${lineRange}`;
    case 'ado':
      return `${repoUrl}?path=/${filePath}&version=GB${branch}${lineStart ? `&line=${lineStart}` : ''}`;
    default:
      return `${repoUrl}/blob/${branch}/${filePath}${lineRange}`;
  }
}

const COLLAPSED_LIMIT = 5;

export function SourceCitations({
  sources,
  repoUrl,
  branch = 'main',
  provider = 'github',
}: SourceCitationsProps) {
  const [expanded, setExpanded] = useState(false);

  if (sources.length === 0) return null;

  const shouldCollapse = sources.length > COLLAPSED_LIMIT;
  const visibleSources = expanded || !shouldCollapse ? sources : sources.slice(0, COLLAPSED_LIMIT);

  return (
    <Box sx={{ mb: 3, p: 2, borderRadius: 1, bgcolor: 'action.hover' }}>
      <Typography
        variant="subtitle2"
        sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 0.5 }}
      >
        <SourceIcon fontSize="small" />
        {sources.length} source file{sources.length !== 1 ? 's' : ''}
      </Typography>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
        {visibleSources.map((src, i) => {
          const label = `${src.file_path}${src.line_start ? `:${src.line_start}${src.line_end ? `-${src.line_end}` : ''}` : ''}`;
          const attributionChip = src.wiki_id ? (
            <Chip
              key={`chip-${i}`}
              size="small"
              label={src.wiki_title ?? src.wiki_id}
              variant="outlined"
              color="primary"
              sx={{ fontSize: '0.65rem', height: 18, mr: 0.5 }}
            />
          ) : null;

          if (repoUrl) {
            const url = buildSourceUrl(
              repoUrl,
              src.file_path,
              branch,
              provider,
              src.line_start,
              src.line_end,
            );
            return (
              <Box key={i} sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 0.25 }}>
                {attributionChip}
                <Link
                  href={url}
                  target="_blank"
                  rel="noopener"
                  underline="hover"
                  sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}
                >
                  {label}
                </Link>
              </Box>
            );
          }

          return (
            <Box key={i} sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 0.25 }}>
              {attributionChip}
              <Typography
                sx={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'text.secondary' }}
              >
                {label}
              </Typography>
            </Box>
          );
        })}
      </Box>

      {shouldCollapse && (
        <Collapse in={!expanded} unmountOnExit>
          <Chip
            label={`Show all ${sources.length} files`}
            size="small"
            onClick={() => setExpanded(true)}
            sx={{ mt: 1 }}
          />
        </Collapse>
      )}

      {shouldCollapse && expanded && (
        <Chip label="Show less" size="small" onClick={() => setExpanded(false)} sx={{ mt: 1 }} />
      )}
    </Box>
  );
}
