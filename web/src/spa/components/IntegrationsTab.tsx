import { useState, useCallback, useEffect } from 'react';
import {
  Box,
  IconButton,
  Paper,
  Snackbar,
  Tab,
  Tabs,
  Typography,
  Alert,
} from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';

interface ToolConfig {
  label: string;
  fileName: string;
  description: string;
  snippet: string;
}

function getBackendUrl(): string {
  if (typeof window !== 'undefined') {
    return window.location.origin;
  }
  return 'http://localhost:8000';
}

function buildConfigs(): ToolConfig[] {
  const url = getBackendUrl();
  const key = 'wikis_YOUR_API_KEY';

  return [
    {
      label: 'Claude Code',
      fileName: '.mcp.json',
      description:
        'Add this to your project root as .mcp.json, or to ~/.claude/mcp.json for global access.',
      snippet: JSON.stringify(
        {
          mcpServers: {
            wikis: {
              type: 'http',
              url: `${url}/mcp`,
              headers: {
                Authorization: `Bearer ${key}`,
              },
            },
          },
        },
        null,
        2,
      ),
    },
    {
      label: 'Cursor',
      fileName: '.cursor/mcp.json',
      description: 'Add this to your project root as .cursor/mcp.json.',
      snippet: JSON.stringify(
        {
          mcpServers: {
            wikis: {
              type: 'http',
              url: `${url}/mcp`,
              headers: {
                Authorization: `Bearer ${key}`,
              },
            },
          },
        },
        null,
        2,
      ),
    },
    {
      label: 'Windsurf',
      fileName: '~/.codeium/windsurf/mcp_config.json',
      description: 'Add this to ~/.codeium/windsurf/mcp_config.json.',
      snippet: JSON.stringify(
        {
          mcpServers: {
            wikis: {
              serverUrl: `${url}/mcp`,
              headers: {
                Authorization: `Bearer ${key}`,
              },
            },
          },
        },
        null,
        2,
      ),
    },
    {
      label: 'VS Code',
      fileName: '.vscode/mcp.json',
      description:
        'Add this to your project as .vscode/mcp.json, or to your User Settings (JSON).',
      snippet: JSON.stringify(
        {
          servers: {
            wikis: {
              type: 'http',
              url: `${url}/mcp`,
              headers: {
                Authorization: `Bearer ${key}`,
              },
            },
          },
        },
        null,
        2,
      ),
    },
  ];
}

function CodeBlock({
  snippet,
  onCopy,
}: {
  snippet: string;
  onCopy: (text: string) => void;
}) {
  return (
    <Box sx={{ position: 'relative' }}>
      <IconButton
        size="small"
        onClick={() => onCopy(snippet)}
        sx={{ position: 'absolute', top: 8, right: 8, color: 'text.secondary' }}
      >
        <ContentCopyIcon fontSize="small" />
      </IconButton>
      <Box
        component="pre"
        sx={{
          m: 0,
          p: 2,
          pr: 5,
          borderRadius: 1,
          bgcolor: 'background.paper',
          border: '1px solid',
          borderColor: 'divider',
          fontSize: '0.8rem',
          lineHeight: 1.6,
          overflow: 'auto',
          fontFamily: 'monospace',
        }}
      >
        {snippet}
      </Box>
    </Box>
  );
}

export function IntegrationsTab() {
  const [activeTab, setActiveTab] = useState(0);
  const [snackbar, setSnackbar] = useState('');
  const [keyCount, setKeyCount] = useState<number | null>(null);

  useEffect(() => {
    fetch('/api/auth/api-key/list', { credentials: 'include' })
      .then((r) => r.json())
      .then((data) => setKeyCount((data.apiKeys ?? []).length))
      .catch(() => setKeyCount(0));
  }, []);

  const configs = buildConfigs();

  const handleCopy = useCallback((text: string) => {
    navigator.clipboard.writeText(text);
    setSnackbar('Copied to clipboard');
  }, []);

  const active = configs[activeTab];

  return (
    <Box>
      <Typography variant="h6" sx={{ mb: 1 }}>
        MCP Integration
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Connect your AI coding tools to Wikis via the{' '}
        <Box component="a" href="https://modelcontextprotocol.io" target="_blank" rel="noopener" sx={{ color: 'primary.main' }}>
          Model Context Protocol
        </Box>
        . This gives your tools access to wiki content, code search, and Q&A across all your indexed repositories.
      </Typography>

      {keyCount === 0 && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          You need an API key to authenticate MCP requests. Generate one in the{' '}
          <strong>API Keys</strong> tab above, copy the full key, and replace{' '}
          <code>wikis_YOUR_API_KEY</code> in the snippet below.
        </Alert>
      )}
      {keyCount !== null && keyCount > 0 && (
        <Alert severity="info" sx={{ mb: 3 }}>
          Replace <code>wikis_YOUR_API_KEY</code> with the full key you copied when creating it.
          The full key is only shown once — if you lost it, generate a new one in the{' '}
          <strong>API Keys</strong> tab.
        </Alert>
      )}

      <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1, textTransform: 'uppercase', fontSize: '0.7rem', letterSpacing: '0.05em' }}>
        Available tools
      </Typography>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mb: 3 }}>
        {[
          { name: 'search_wikis', desc: 'List all indexed wikis and find a wiki_id by repo name or URL. Start every session here.' },
          { name: 'list_wiki_pages', desc: 'Return all page IDs, titles, and sections for a wiki. Use to navigate structure before reading.' },
          { name: 'get_wiki_page', desc: 'Read a page\'s full markdown content. Supports offset/limit for paginating large pages.' },
          { name: 'ask_codebase', desc: 'Ask a natural language question and get an AI answer grounded in the wiki. Fast, single-shot.' },
          { name: 'research_codebase', desc: 'Run a deep multi-step research agent across the codebase. Slower but thorough — best for complex architectural questions.' },
          { name: 'map_codebase', desc: 'Generate a hierarchical call-tree showing which files and functions are involved in a feature or flow. Best for tracing request paths and understanding how things are wired together.' },
        ].map((tool) => (
          <Paper
            key={tool.name}
            variant="outlined"
            sx={{ px: 2, py: 1, borderRadius: 1, display: 'flex', alignItems: 'center', gap: 2 }}
          >
            <Typography variant="caption" sx={{ fontFamily: 'monospace', fontWeight: 700, whiteSpace: 'nowrap', color: 'text.primary', width: 180, flexShrink: 0 }}>
              {tool.name}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {tool.desc}
            </Typography>
          </Paper>
        ))}
      </Box>

      <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1, textTransform: 'uppercase', fontSize: '0.7rem', letterSpacing: '0.05em' }}>
        Setup
      </Typography>
      <Paper variant="outlined" sx={{ borderRadius: 1, overflow: 'hidden' }}>
        <Tabs
          value={activeTab}
          onChange={(_, v) => setActiveTab(v)}
          sx={{
            borderBottom: '1px solid',
            borderColor: 'divider',
            minHeight: 40,
            '& .MuiTab-root': { minHeight: 40, textTransform: 'none', fontSize: '0.85rem' },
          }}
        >
          {configs.map((c, i) => (
            <Tab key={i} label={c.label} />
          ))}
        </Tabs>

        <Box sx={{ p: 2.5 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
            {active.description}
          </Typography>
          <Typography
            variant="caption"
            sx={{ fontFamily: 'monospace', color: 'text.disabled', display: 'block', mb: 1.5 }}
          >
            {active.fileName}
          </Typography>
          <CodeBlock snippet={active.snippet} onCopy={handleCopy} />
        </Box>
      </Paper>

      <Snackbar
        open={!!snackbar}
        autoHideDuration={2000}
        onClose={() => setSnackbar('')}
        message={snackbar}
      />
    </Box>
  );
}
