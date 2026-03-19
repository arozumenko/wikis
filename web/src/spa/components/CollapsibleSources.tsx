import { Box, Link, Typography } from '@mui/material';

interface SourceFile {
  file_path: string;
  line_start?: number | null;
  line_end?: number | null;
}

interface CollapsibleSourcesProps {
  sources: SourceFile[];
  repoUrl?: string;
  branch?: string;
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

export function CollapsibleSources({ sources, repoUrl, branch = 'main' }: CollapsibleSourcesProps) {
  if (sources.length === 0) return null;

  return (
    <Box
      component="details"
      sx={{
        my: 3,
        p: 0,
        '&[open] summary': { mb: 1 },
      }}
    >
      <Box
        component="summary"
        sx={{
          cursor: 'pointer',
          color: 'text.secondary',
          fontSize: '0.8rem',
          fontWeight: 500,
          userSelect: 'none',
          '&::-webkit-details-marker': { display: 'none' },
          '&::marker': { content: '""' },
          display: 'flex',
          alignItems: 'center',
          gap: 0.5,
          '&::before': {
            content: '"▸"',
            fontSize: '0.7rem',
            transition: 'transform 0.15s',
          },
          '[open] > &::before': {
            transform: 'rotate(90deg)',
          },
        }}
      >
        Relevant source files ({sources.length})
      </Box>
      <Box sx={{ pl: 2 }}>
        {sources.map((src, i) => {
          const label = `${src.file_path}${src.line_start ? `:${src.line_start}` : ''}`;
          return repoUrl ? (
            <Link
              key={i}
              href={buildUrl(repoUrl, src.file_path, branch, src.line_start)}
              target="_blank"
              rel="noopener"
              display="block"
              underline="hover"
              sx={{ fontFamily: 'monospace', fontSize: '0.78rem', py: 0.2 }}
            >
              {label}
            </Link>
          ) : (
            <Typography
              key={i}
              sx={{
                fontFamily: 'monospace',
                fontSize: '0.78rem',
                color: 'text.secondary',
                py: 0.2,
              }}
            >
              {label}
            </Typography>
          );
        })}
      </Box>
    </Box>
  );
}
