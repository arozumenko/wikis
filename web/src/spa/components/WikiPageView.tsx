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

function stripFrontmatter(raw: string): string {
  if (!raw.startsWith('---')) return raw;
  const end = raw.indexOf('\n---', 3);
  if (end === -1) return raw;
  return raw.slice(end + 4).replace(/^\r?\n/, '');
}

export function WikiPageView({ content, mode = 'dark', onNavigate }: WikiPageViewProps) {
  // Callout type → color palette
  const calloutColors: Record<string, { border: string; bg: string; label: string }> = {
    abstract: { border: '#a78bfa', bg: 'rgba(167,139,250,0.08)', label: '📋' },
    info:     { border: '#60a5fa', bg: 'rgba(96,165,250,0.08)',  label: 'ℹ️' },
    tip:      { border: '#34d399', bg: 'rgba(52,211,153,0.08)',  label: '💡' },
    warning:  { border: '#fbbf24', bg: 'rgba(251,191,36,0.08)', label: '⚠️' },
    example:  { border: '#22d3ee', bg: 'rgba(34,211,238,0.08)', label: '📌' },
    danger:   { border: '#f87171', bg: 'rgba(248,113,113,0.08)', label: '🔴' },
    note:     { border: '#60a5fa', bg: 'rgba(96,165,250,0.08)',  label: '📝' },
    success:  { border: '#34d399', bg: 'rgba(52,211,153,0.08)',  label: '✅' },
    question: { border: '#fbbf24', bg: 'rgba(251,191,36,0.08)', label: '❓' },
    bug:      { border: '#f87171', bg: 'rgba(248,113,113,0.08)', label: '🐛' },
    quote:    { border: '#94a3b8', bg: 'rgba(148,163,184,0.08)', label: '💬' },
  };

  const components: Components = {
    blockquote({ children }) {
      // Detect Obsidian callout: first text content starts with [!type]
      const firstText = extractText(children);
      const match = firstText.match(/^\[!(\w+)\][ \t]*(.*)/);
      if (match) {
        const type = match[1].toLowerCase();
        const inlineTitle = match[2].trim();
        const palette = calloutColors[type] ?? { border: '#94a3b8', bg: 'rgba(148,163,184,0.08)', label: '📄' };
        // Strip the [!type] marker line from children so it's not rendered twice
        const cleanedChildren = (() => {
          const arr = Array.isArray(children) ? children : [children];
          return arr.map((child, idx) => {
            if (idx !== 0) return child;
            if (typeof child === 'string') return child.replace(/^\[!\w+\][^\n]*\n?/, '');
            if (child && typeof child === 'object' && 'props' in child) {
              const inner = child.props.children;
              const firstChild = Array.isArray(inner) ? inner[0] : inner;
              if (typeof firstChild === 'string') {
                const rest = firstChild.replace(/^\[!\w+\][^\n]*\n?/, '');
                const newInner = Array.isArray(inner) ? [rest, ...inner.slice(1)] : rest;
                return { ...child, props: { ...child.props, children: newInner } };
              }
            }
            return child;
          });
        })();
        return (
          <Box
            component="div"
            sx={{
              borderLeft: `4px solid ${palette.border}`,
              background: palette.bg,
              borderRadius: '0 6px 6px 0',
              px: 2,
              py: 1,
              my: 2,
            }}
          >
            <Box sx={{ fontWeight: 700, color: palette.border, mb: 0.5, fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              {palette.label} {type}{inlineTitle ? ` — ${inlineTitle}` : ''}
            </Box>
            <Box sx={{ '& p:last-child': { mb: 0 } }}>{cleanedChildren}</Box>
          </Box>
        );
      }
      return (
        <Box
          component="blockquote"
          sx={{ borderLeft: '3px solid rgba(255,255,255,0.2)', pl: 2, my: 1, color: 'text.secondary', fontStyle: 'italic' }}
        >
          {children}
        </Box>
      );
    },
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
        {stripFrontmatter(content)}
      </ReactMarkdown>
    </Box>
  );
}
