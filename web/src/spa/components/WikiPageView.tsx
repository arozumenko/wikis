import { Box, Chip, Collapse, IconButton, Tooltip, Typography } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import { ShareButton } from './ShareButton';
import { ExportButton } from './ExportButton';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeSlug from 'rehype-slug';
import { visit } from 'unist-util-visit';
import { MermaidDiagram } from './MermaidDiagram';
import { CodeBlock } from './CodeBlock';
import type { Components } from 'react-markdown';
import React, { useState } from 'react';

/** Remark plugin: transforms Obsidian callout blockquotes into annotated hast nodes.
 *  Detects `> [!type] title` pattern, attaches data-callout-* hProperties to the
 *  blockquote, and removes the marker line from the AST so children are body-only.
 */
function remarkCallout() {
  return (tree: any) => {
    visit(tree, 'blockquote', (node: any) => {
      if (!node.children?.length) return;
      const firstPara = node.children[0];
      if (firstPara.type !== 'paragraph' || !firstPara.children?.length) return;
      const firstNode = firstPara.children[0];
      if (firstNode.type !== 'text') return;

      const match = firstNode.value.match(/^\[!(\w+)\][ \t]*(.*)/);
      if (!match) return;

      // Attach type and title to blockquote hast properties
      node.data = node.data ?? {};
      node.data.hProperties = {
        ...node.data.hProperties,
        'data-callout-type': match[1].toLowerCase(),
        'data-callout-title': match[2].trim(),
      };

      // Remove the marker line from the AST
      const newlineIdx = firstNode.value.indexOf('\n');
      if (newlineIdx >= 0) {
        // Marker and body in same text node — keep only the part after the newline
        firstNode.value = firstNode.value.slice(newlineIdx + 1);
        if (!firstNode.value && firstPara.children.length === 1) node.children.shift();
      } else {
        // Check for a hard break node after the marker within the same paragraph
        const breakIdx = firstPara.children.findIndex((n: any, i: number) => i > 0 && n.type === 'break');
        if (breakIdx > 0) {
          firstPara.children = firstPara.children.slice(breakIdx + 1);
          if (!firstPara.children.length) node.children.shift();
        } else {
          // Whole first paragraph is just the marker — remove it
          node.children.shift();
        }
      }
    });
  };
}

interface WikiPageViewProps {
  content: string;
  mode?: 'light' | 'dark';
  onNavigate?: (pageId: string) => void;
  pages?: { id: string; title: string }[];
  wikiId?: string;
  wikiTitle?: string;
  isWikiComplete?: boolean;
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

interface FrontmatterMeta {
  title?: string;
  description?: string;
  type?: string;
  section?: string;
  wiki?: string;
  repository?: string;
  branch?: string;
  created?: string;
  tags?: string[];
  aliases?: string[];
  key_files?: string[];
  symbols?: string[];
  source_files?: string[];
  [key: string]: unknown;
}

function parseFrontmatter(raw: string): { meta: FrontmatterMeta; content: string } {
  if (!raw.startsWith('---')) return { meta: {}, content: raw };
  const end = raw.indexOf('\n---', 3);
  if (end === -1) return { meta: {}, content: raw };
  const yamlStr = raw.slice(4, end);
  const content = raw.slice(end + 4).replace(/^\r?\n/, '');

  const meta: FrontmatterMeta = {};
  const lines = yamlStr.split('\n');
  let currentKey: string | null = null;
  let currentList: string[] | null = null;

  for (const line of lines) {
    const listItem = line.match(/^[ \t]+-[ \t]+(.+)/);
    if (listItem && currentKey) {
      if (!currentList) {
        currentList = [];
        (meta as Record<string, unknown>)[currentKey] = currentList;
      }
      currentList.push(listItem[1].trim().replace(/^["']|["']$/g, ''));
      continue;
    }
    const kv = line.match(/^([a-zA-Z_][a-zA-Z0-9_-]*)\s*:\s*(.*)$/);
    if (kv) {
      currentList = null;
      currentKey = kv[1];
      const val = kv[2].trim().replace(/^["']|["']$/g, '');
      if (val === '') {
        currentList = [];
        (meta as Record<string, unknown>)[currentKey] = currentList;
      } else {
        (meta as Record<string, unknown>)[currentKey] = val;
      }
    }
  }
  return { meta, content };
}

/** Convert [[Target|Label]] and [[Target]] wikilinks to [Label](wiki:Target) markdown links. */
function processWikilinks(content: string): string {
  return content.replace(/\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]/g, (_, target, label) => {
    const display = (label ?? target).trim();
    return `[${display}](wiki:${encodeURIComponent(target.trim())})`;
  });
}

function WikiProperties({ meta, mode }: { meta: FrontmatterMeta; mode: 'light' | 'dark' }) {
  const [open, setOpen] = useState(false);

  const hasMeta = meta.description || meta.section || meta.repository || (meta.tags && meta.tags.length > 0) || (meta.key_files && meta.key_files.length > 0);
  if (!hasMeta) return null;

  // Filter tags to surface-friendly ones (skip long wiki/ prefixed tags)
  const displayTags = (meta.tags ?? []).filter(
    (t) => !t.startsWith('wiki/') && !t.startsWith('section/')
  );

  const dim = mode === 'dark' ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.04)';
  const border = mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';

  return (
    <Box sx={{ mb: 3, border: `1px solid ${border}`, borderRadius: 2, overflow: 'hidden', fontSize: '0.85rem' }}>
      <Box
        onClick={() => setOpen((o) => !o)}
        sx={{
          display: 'flex', alignItems: 'center', gap: 1, px: 2, py: 1,
          bgcolor: dim, cursor: 'pointer', userSelect: 'none',
          '&:hover': { bgcolor: mode === 'dark' ? 'rgba(255,255,255,0.09)' : 'rgba(0,0,0,0.07)' },
        }}
      >
        <span style={{ fontSize: '0.75rem', opacity: 0.6 }}>{open ? '▾' : '▸'}</span>
        <Typography variant="caption" sx={{ fontWeight: 600, opacity: 0.7, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          Properties
        </Typography>
        {meta.type && (
          <Chip label={meta.type} size="small" sx={{ height: 18, fontSize: '0.7rem', ml: 'auto', opacity: 0.7 }} />
        )}
      </Box>
      <Collapse in={open}>
        <Box sx={{ px: 2, py: 1.5, display: 'flex', flexDirection: 'column', gap: 1 }}>
          {meta.description && (
            <Box>
              <Typography variant="caption" sx={{ opacity: 0.5, display: 'block', mb: 0.25 }}>description</Typography>
              <Typography variant="body2" sx={{ lineHeight: 1.5 }}>{meta.description}</Typography>
            </Box>
          )}
          {meta.section && (
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              <Typography variant="caption" sx={{ opacity: 0.5, minWidth: 80 }}>section</Typography>
              <Typography variant="body2">{meta.section}</Typography>
            </Box>
          )}
          {meta.repository && (
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              <Typography variant="caption" sx={{ opacity: 0.5, minWidth: 80 }}>repository</Typography>
              <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                {meta.repository}{meta.branch ? `@${meta.branch}` : ''}
              </Typography>
            </Box>
          )}
          {displayTags.length > 0 && (
            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', alignItems: 'center' }}>
              <Typography variant="caption" sx={{ opacity: 0.5, minWidth: 80 }}>tags</Typography>
              {displayTags.map((t) => (
                <Chip key={t} label={t.replace(/^type\/|^lang\//, '')} size="small"
                  sx={{ height: 18, fontSize: '0.7rem', bgcolor: dim }} />
              ))}
            </Box>
          )}
          {meta.key_files && meta.key_files.length > 0 && (
            <Box>
              <Typography variant="caption" sx={{ opacity: 0.5, display: 'block', mb: 0.25 }}>key files</Typography>
              <Box component="ul" sx={{ m: 0, pl: 2, '& li': { fontFamily: 'monospace', fontSize: '0.75rem', opacity: 0.8 } }}>
                {meta.key_files.map((f) => <li key={f}>{f}</li>)}
              </Box>
            </Box>
          )}
        </Box>
      </Collapse>
    </Box>
  );
}

export function WikiPageView({ content, mode = 'dark', onNavigate, pages = [], wikiId, wikiTitle, isWikiComplete }: WikiPageViewProps) {
  const { meta, content: body } = parseFrontmatter(content);
  const processedBody = processWikilinks(body);
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
    // Suppress the react-markdown <pre> wrapper — MermaidDiagram and CodeBlock own their card styling
    pre({ children }) {
      return <>{children}</>;
    },
    blockquote({ children, node }) {
      const type = (node as any)?.properties?.['data-callout-type'] as string | undefined;
      const title = (node as any)?.properties?.['data-callout-title'] as string | undefined;
      if (type) {
        const palette = calloutColors[type] ?? { border: '#94a3b8', bg: 'rgba(148,163,184,0.08)', label: '📄' };
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
              {palette.label} {type}{title ? ` — ${title}` : ''}
            </Box>
            <Box sx={{ '& p:last-child': { mb: 0 } }}>{children}</Box>
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
      // Wikilinks resolved to wiki: protocol
      if (href && href.startsWith('wiki:') && onNavigate) {
        const title = decodeURIComponent(href.slice(5));
        const page = pages.find(
          (p) => p.title === title || p.title.toLowerCase() === title.toLowerCase()
        );
        if (page) {
          return (
            <a
              href={`#${page.id}`}
              onClick={(e) => { e.preventDefault(); onNavigate(page.id); }}
              style={{ cursor: 'pointer' }}
              {...props}
            >
              {children}
            </a>
          );
        }
        // Unresolved wikilink — show as dimmed non-link
        return (
          <span style={{ opacity: 0.5, textDecoration: 'underline dotted', cursor: 'not-allowed' }}>
            {children}
          </span>
        );
      }
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
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 1, mb: 1 }}>
        {wikiId && (
          <ExportButton
            wikiId={wikiId}
            wikiTitle={wikiTitle ?? ''}
            isComplete={isWikiComplete ?? false}
          />
        )}
        <ShareButton />
      </Box>
      <WikiProperties meta={meta} mode={mode} />
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkCallout]}
        rehypePlugins={[rehypeSlug, rehypeHighlight]}
        components={components}
        urlTransform={(url) => (url.startsWith('javascript:') ? '' : url)}
      >
        {processedBody}
      </ReactMarkdown>
    </Box>
  );
}
