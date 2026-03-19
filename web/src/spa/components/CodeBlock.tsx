import { Box, IconButton, Tooltip } from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import { ReactNode, useCallback, useState } from 'react';

interface CodeBlockProps {
  /** Plain text for copy-to-clipboard */
  code: string;
  language?: string;
  /** Pre-highlighted React nodes from rehype-highlight (if available) */
  highlightedChildren?: ReactNode;
}

export function CodeBlock({ code, language, highlightedChildren }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [code]);

  return (
    <Box sx={{ position: 'relative', my: 2 }}>
      <Tooltip title={copied ? 'Copied!' : 'Copy code'} placement="left">
        <IconButton
          size="small"
          onClick={handleCopy}
          sx={{
            position: 'absolute',
            top: 8,
            right: 8,
            zIndex: 1,
            opacity: 0.6,
            '&:hover': { opacity: 1 },
          }}
        >
          <ContentCopyIcon fontSize="small" />
        </IconButton>
      </Tooltip>
      <pre style={{ margin: 0, overflow: 'auto' }}>
        <code className={language ? `hljs language-${language}` : 'hljs'}>
          {highlightedChildren ?? code}
        </code>
      </pre>
    </Box>
  );
}
