import { useEffect, useRef, useState } from 'react';
import { Alert, Box, Typography } from '@mui/material';
import mermaid from 'mermaid';

interface MermaidDiagramProps {
  chart: string;
  mode?: 'light' | 'dark';
}

let mermaidInitialized = false;

export function MermaidDiagram({ chart, mode = 'dark' }: MermaidDiagramProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: false,
      theme: mode === 'dark' ? 'dark' : 'default',
      securityLevel: 'strict',
      fontFamily: '"Inter", Roboto, Arial, sans-serif',
      fontSize: 14,
      logLevel: 'fatal',
    });
    mermaidInitialized = true;
  }, [mode]);

  useEffect(() => {
    if (!chart || !mermaidInitialized) return;

    const render = async () => {
      try {
        setError(null);
        await mermaid.parse(chart);
        const id = `mermaid-${Math.random().toString(36).slice(2, 11)}`;
        const { svg: rendered } = await mermaid.render(id, chart);
        setSvg(rendered);
      } catch (err) {
        setSvg(null);
        setError(err instanceof Error ? err.message : 'Failed to render diagram');
      }
    };

    render();
  }, [chart, mode]);

  if (error) {
    return (
      <Alert severity="error" sx={{ my: 2 }}>
        <Typography variant="body2" sx={{ fontWeight: 500 }}>
          Failed to render Mermaid diagram
        </Typography>
        <Typography
          variant="body2"
          component="pre"
          sx={{ fontFamily: 'monospace', fontSize: '11px', whiteSpace: 'pre-wrap', m: 0, mt: 0.5 }}
        >
          {error}
        </Typography>
      </Alert>
    );
  }

  if (!svg) return null;

  return (
    <Box
      ref={containerRef}
      sx={{
        my: 3,
        p: 2,
        borderRadius: 2,
        bgcolor: mode === 'dark' ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)',
        border: '1px solid',
        borderColor: 'divider',
        overflow: 'auto',
        '& svg': { display: 'block', maxWidth: '100%', height: 'auto' },
      }}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
