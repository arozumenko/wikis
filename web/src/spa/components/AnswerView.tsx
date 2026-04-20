import { Box, CircularProgress, Typography } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { MermaidDiagram } from './MermaidDiagram';
import { CodeBlock } from './CodeBlock';
import type { Components } from 'react-markdown';

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

interface AnswerViewProps {
  question: string;
  answer: string | null;
  loading: boolean;
  mode?: 'light' | 'dark';
}

export function AnswerView({ question, answer, loading, mode = 'dark' }: AnswerViewProps) {
  const components: Components = {
    // Strip the outer <pre> that rehype-highlight wraps around <code>
    // to avoid double-card rendering (CodeBlock already provides its own <pre>)
    pre({ children }) {
      return <>{children}</>;
    },
    code({ className, children }) {
      const match = /language-(\w+)/.exec(className ?? '');
      const language = match?.[1];
      const plainText = extractText(children).replace(/\n$/, '');

      if (language === 'mermaid') {
        return <MermaidDiagram chart={plainText} mode={mode} />;
      }

      if (plainText.includes('\n') || className?.includes('hljs')) {
        return <CodeBlock code={plainText} language={language} highlightedChildren={children} />;
      }

      return <code className={className}>{children}</code>;
    },
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', pt: 4, pb: 12, px: { xs: 3, md: 5 } }}>
      {/* Question */}
      <Typography
        variant="h5"
        sx={{
          fontFamily: '"Playfair Display", Georgia, serif',
          fontWeight: 600,
          mb: 4,
          pb: 3,
          borderBottom: '1px solid',
          borderColor: 'divider',
        }}
      >
        {question}
      </Typography>

      {/* Loading — spinner only when no answer text yet */}
      {loading && !answer && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, py: 4 }}>
          <CircularProgress size={20} />
          <Typography color="text.secondary">Thinking...</Typography>
        </Box>
      )}

      {/* Answer — renders progressively as chunks arrive */}
      {answer && (
        <Box
          sx={{
            '& h1': { mt: 0, mb: 2, fontSize: '1.5rem' },
            '& h2': { mt: 4, mb: 1.5, fontSize: '1.25rem' },
            '& p': { mb: 2, lineHeight: 1.8 },
            '& ul, & ol': { mb: 2, pl: 3 },
            '& li': { mb: 0.5, lineHeight: 1.7 },
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
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: '0.85em',
            },
            '& :not(pre) > code': {
              bgcolor: mode === 'dark' ? 'rgba(255, 107, 74, 0.1)' : 'rgba(255, 107, 74, 0.08)',
              color: 'primary.main',
              px: 0.75,
              py: 0.25,
              borderRadius: '4px',
            },
            '& a': { color: 'primary.main' },
            '& blockquote': {
              borderLeft: '4px solid',
              borderColor: 'primary.main',
              pl: 2,
              ml: 0,
              color: 'text.secondary',
              fontStyle: 'italic',
            },
          }}
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={components}
          >
            {answer}
          </ReactMarkdown>
          {/* Blinking cursor while answer is still streaming */}
          {loading && (
            <Box
              component="span"
              sx={{
                display: 'inline-block',
                width: 8,
                height: '1.2em',
                bgcolor: 'primary.main',
                borderRadius: '2px',
                ml: 0.5,
                verticalAlign: 'text-bottom',
                animation: 'blink 1s step-end infinite',
                '@keyframes blink': { '50%': { opacity: 0 } },
              }}
            />
          )}
        </Box>
      )}
    </Box>
  );
}
