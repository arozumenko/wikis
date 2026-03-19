import { Box, Chip, Typography } from '@mui/material';
import DescriptionIcon from '@mui/icons-material/Description';

interface SourceRef {
  file_path: string;
  line_start?: number | null;
  line_end?: number | null;
}

interface CitationChipsProps {
  sources: SourceRef[];
  repoUrl?: string;
  branch?: string;
}

function isLocalRepo(repoUrl?: string): boolean {
  return !!repoUrl && (repoUrl.startsWith('/') || repoUrl.startsWith('file://'));
}

function buildUrl(
  repoUrl: string,
  filePath: string,
  branch: string,
  lineStart?: number | null,
): string {
  const line = lineStart ? `#L${lineStart}` : '';
  return `${repoUrl}/blob/${branch}/${filePath}${line}`;
}

function shortPath(filePath: string): string {
  const parts = filePath.split('/');
  if (parts.length <= 2) return filePath;
  return `.../${parts.slice(-2).join('/')}`;
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
            repoUrl && !isLocalRepo(repoUrl)
              ? buildUrl(repoUrl, src.file_path, branch, src.line_start)
              : undefined;

          return (
            <Chip
              key={i}
              icon={<DescriptionIcon sx={{ fontSize: 14 }} />}
              label={label}
              size="small"
              variant="outlined"
              component={url ? 'a' : 'span'}
              href={url}
              target="_blank"
              rel="noopener"
              clickable={!!url}
              sx={{
                fontFamily: 'monospace',
                fontSize: '0.72rem',
                height: 26,
                '& .MuiChip-icon': { ml: 0.5 },
              }}
            />
          );
        })}
      </Box>
    </Box>
  );
}
