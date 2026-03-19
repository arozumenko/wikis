import { useState } from 'react';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  IconButton,
  Link,
  Tooltip,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';

interface Source {
  file_path: string;
  line_start?: number | null;
  line_end?: number | null;
  snippet?: string | null;
}

interface SourcePanelProps {
  sources: Source[];
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

export function SourcePanel({ sources, repoUrl, branch = 'main' }: SourcePanelProps) {
  const [expanded, setExpanded] = useState<string | false>(sources[0]?.file_path ?? false);

  if (sources.length === 0) {
    return (
      <Box sx={{ p: 3, color: 'text.secondary', textAlign: 'center' }}>
        <Typography variant="body2">No source references</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ overflow: 'auto', height: '100%' }}>
      <Typography
        variant="caption"
        sx={{
          px: 2,
          pt: 2,
          pb: 1,
          display: 'block',
          fontWeight: 600,
          color: 'text.secondary',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          fontSize: '0.65rem',
        }}
      >
        Sources ({sources.length})
      </Typography>

      {sources.map((src, i) => {
        const label = `${src.file_path}${src.line_start ? `:${src.line_start}${src.line_end ? `-${src.line_end}` : ''}` : ''}`;
        const url = repoUrl ? buildUrl(repoUrl, src.file_path, branch, src.line_start) : undefined;

        return (
          <Accordion
            key={i}
            expanded={expanded === src.file_path}
            onChange={(_, isExpanded) => setExpanded(isExpanded ? src.file_path : false)}
            disableGutters
            elevation={0}
            sx={{ '&:before': { display: 'none' }, bgcolor: 'transparent' }}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon sx={{ fontSize: 16 }} />}
              sx={{ minHeight: 36, px: 2, '& .MuiAccordionSummary-content': { my: 0.5 } }}
            >
              <Typography
                variant="caption"
                sx={{ fontFamily: 'monospace', fontSize: '0.75rem', color: 'text.secondary' }}
              >
                {label}
              </Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ px: 2, pt: 0, pb: 1 }}>
              {src.snippet ? (
                <Box sx={{ position: 'relative' }}>
                  <pre
                    style={{ margin: 0, overflow: 'auto', fontSize: '0.78rem', lineHeight: 1.5 }}
                  >
                    <code>{src.snippet}</code>
                  </pre>
                  <Box sx={{ position: 'absolute', top: 0, right: 0, display: 'flex', gap: 0.25 }}>
                    <Tooltip title="Copy">
                      <IconButton
                        size="small"
                        onClick={() => navigator.clipboard.writeText(src.snippet!)}
                        sx={{ opacity: 0.5, '&:hover': { opacity: 1 } }}
                      >
                        <ContentCopyIcon sx={{ fontSize: 14 }} />
                      </IconButton>
                    </Tooltip>
                    {url && (
                      <Tooltip title="Open in GitHub">
                        <IconButton
                          size="small"
                          component="a"
                          href={url}
                          target="_blank"
                          rel="noopener"
                          sx={{ opacity: 0.5, '&:hover': { opacity: 1 } }}
                        >
                          <OpenInNewIcon sx={{ fontSize: 14 }} />
                        </IconButton>
                      </Tooltip>
                    )}
                  </Box>
                </Box>
              ) : url ? (
                <Link
                  href={url}
                  target="_blank"
                  rel="noopener"
                  underline="hover"
                  variant="caption"
                  sx={{ fontFamily: 'monospace' }}
                >
                  View on GitHub
                </Link>
              ) : (
                <Typography variant="caption" color="text.secondary">
                  No snippet available
                </Typography>
              )}
            </AccordionDetails>
          </Accordion>
        );
      })}
    </Box>
  );
}
