import { Box } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import { ShareButton } from './ShareButton';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { MermaidDiagram } from './MermaidDiagram';
import { CodeBlock } from './CodeBlock';
import type { Components } from 'react-markdown';

interface WikiPageViewProps {
  content: string;
  mode?: 'light' | 'dark';
  onNavigate?: (pageId: string) => void;
}

/**
 * Recursively extract plain text from React children.
 * Handles strings, numbers, arrays, and React elements with nested children.
 */
function extractText(node: React.ReactNode): string {
  if (node == null || typeof node === 'boolean') return '';
  if (typeof node === 'string') return node;
  if (typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(extractText).join('');
  if (typeof node === 'object' && 'props' in node) {
    return extractText(node.props.children);
  }
  return '';
}

export function WikiPageView({ content, mode = 'dark', onNavigate }: WikiPageViewProps) {
  const components: Components = {
    a({ href, children, ...props }) {
      // Intercept internal .md links
      if (href && href.endsWith('.md') && !href.startsWith('http') && onNavigate) {
        const pageId = href.replace(/^\.\//, '').replace(/\.md$/, '');
        return (
          <a
            href={`#${pageId}`}
            onClick={(e) => {
              e.preventDefault();
              onNavigate(pageId);
            }}
            style={{ cursor: 'pointer' }}
            {...props}
          >
            {children}
          </a>
        );
      }
      // Fragment links scroll within the page; external links open in new tab
      if (href && href.startsWith('#')) {
        return (
          <a href={href} {...props}>
            {children}
          </a>
        );
      }
      return (
        <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
          {children}
        </a>
      );
    },
    code({ className, children, ...props }) {
      const match = /language-(\w+)/.exec(className ?? '');
      const language = match?.[1];
      const plainText = extractText(children).replace(/\n$/, '');

      // Mermaid blocks get rendered as diagrams
      if (language === 'mermaid') {
        return <MermaidDiagram chart={plainText} mode={mode} />;
      }

      // Multi-line code blocks get the CodeBlock component
      // Pass both plain text (for copy) and React children (for highlighted display)
      if (plainText.includes('\n') || className?.includes('hljs')) {
        return <CodeBlock code={plainText} language={language} highlightedChildren={children} />;
      }

      // Inline code
      return (
        <code className={className} {...props}>
          {children}
        </code>
      );
    },
  };

  return (
    <Box
      sx={{
        flex: 1,
        overflow: 'auto',
        scrollbarWidth: 'thin',
        scrollbarColor: 'rgba(255,255,255,0.15) transparent',
        '&::-webkit-scrollbar': { width: 6 },
        '&::-webkit-scrollbar-track': { background: 'transparent' },
        '&::-webkit-scrollbar-thumb': {
          background: 'rgba(255,255,255,0.15)',
          borderRadius: 3,
          '&:hover': { background: 'rgba(255,255,255,0.25)' },
        },
        px: { xs: 2, md: 4 },
        pt: 4,
        pb: 12,
        '& h1': {
          fontFamily: '"Playfair Display", Georgia, serif',
          mt: 0,
          mb: 3,
          fontSize: '2rem',
          letterSpacing: '-0.02em',
        },
        '& h2': {
          fontFamily: '"Playfair Display", Georgia, serif',
          mt: 6,
          mb: 2,
          pt: 4,
          fontSize: '1.5rem',
          letterSpacing: '-0.01em',
          borderTop: '1px solid',
          borderColor: 'divider',
        },
        '& h3': {
          fontFamily: '"Playfair Display", Georgia, serif',
          mt: 4,
          mb: 1.5,
          fontSize: '1.25rem',
        },
        '& p': { mb: 2.5, lineHeight: 1.8 },
        '& ul, & ol': { mb: 2.5, pl: 3 },
        '& li': { mb: 0.75, lineHeight: 1.7 },
        '& table': {
          width: '100%',
          borderCollapse: 'collapse',
          mb: 3,
          borderRadius: '8px',
          overflow: 'hidden',
          '& th, & td': {
            border: '1px solid',
            borderColor: 'divider',
            px: 2,
            py: 1.25,
            textAlign: 'left',
          },
          '& th': { fontWeight: 600, bgcolor: 'action.hover' },
          '& tr:nth-of-type(even) td': {
            bgcolor: mode === 'dark' ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)',
          },
        },
        '& blockquote': {
          borderLeft: '4px solid',
          borderColor: 'primary.main',
          pl: 2.5,
          ml: 0,
          my: 3,
          color: 'text.secondary',
          fontStyle: 'italic',
        },
        '& pre': {
          borderRadius: '12px',
          overflow: 'auto',
          bgcolor: mode === 'dark' ? '#1A2332' : '#F1F5F9',
          p: 2.5,
          my: 3,
          border: '1px solid',
          borderColor: 'divider',
        },
        '& code': {
          fontFamily: '"JetBrains Mono", "Fira Code", monospace',
          fontSize: '0.85em',
        },
        '& :not(pre) > code': {
          bgcolor: mode === 'dark' ? 'rgba(255, 107, 74, 0.1)' : 'rgba(255, 107, 74, 0.08)',
          color: 'primary.main',
          px: 0.75,
          py: 0.25,
          borderRadius: '4px',
          fontWeight: 500,
        },
        '& a': {
          color: 'primary.main',
          textDecoration: 'none',
          '&:hover': { textDecoration: 'underline' },
        },
        '& img': { maxWidth: '100%', borderRadius: '8px' },
        '& hr': { border: 'none', borderTop: '1px solid', borderColor: 'divider', my: 4 },
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
        <ShareButton />
      </Box>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </Box>
  );
}
