import { useCallback, useEffect, useRef, useState } from 'react';
import { Box, List, ListItemButton, ListItemText, Typography } from '@mui/material';

interface Heading {
  id: string;
  text: string;
  level: number;
}

interface OnThisPageProps {
  contentRef: React.RefObject<HTMLDivElement | null>;
}

export function OnThisPage({ contentRef }: OnThisPageProps) {
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);

  // Extract headings from rendered content
  useEffect(() => {
    const el = contentRef.current;
    if (!el) return;

    const extract = () => {
      const nodes = el.querySelectorAll('h2, h3');
      const result: Heading[] = [];
      nodes.forEach((node) => {
        const text = node.textContent?.trim() ?? '';
        if (!text) return;
        let id = node.id;
        if (!id) {
          id = text
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '-')
            .replace(/^-|-$/g, '');
          node.id = id;
        }
        result.push({ id, text, level: node.tagName === 'H2' ? 2 : 3 });
      });
      setHeadings(result);
    };

    const observer = new MutationObserver(extract);
    observer.observe(el, { childList: true, subtree: true });
    extract();

    return () => observer.disconnect();
  }, [contentRef]);

  // Track active heading with IntersectionObserver
  useEffect(() => {
    if (headings.length === 0) return;
    observerRef.current?.disconnect();

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible.length > 0) {
          setActiveId(visible[0].target.id);
        }
      },
      { rootMargin: '-80px 0px -60% 0px', threshold: 0 },
    );

    headings.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    observerRef.current = observer;
    return () => observer.disconnect();
  }, [headings]);

  const handleClick = useCallback((id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      window.history.replaceState(null, '', `#${id}`);
    }
  }, []);

  if (headings.length <= 1) return null;

  return (
    <Box
      sx={{
        width: 200,
        flexShrink: 0,
        position: 'sticky',
        top: 80,
        maxHeight: 'calc(100vh - 100px)',
        overflow: 'auto',
        display: { xs: 'none', lg: 'block' },
      }}
    >
      <Typography
        variant="caption"
        sx={{
          px: 1.5,
          pb: 0.5,
          fontWeight: 700,
          color: 'text.secondary',
          textTransform: 'uppercase',
        }}
      >
        On this page
      </Typography>
      <List dense disablePadding>
        {headings.map((h) => (
          <ListItemButton
            key={h.id}
            onClick={() => handleClick(h.id)}
            sx={{
              py: 0.25,
              pl: h.level === 3 ? 3 : 1.5,
              borderLeft: '2px solid',
              borderColor: activeId === h.id ? 'primary.main' : 'transparent',
            }}
          >
            <ListItemText
              primary={h.text}
              primaryTypographyProps={{
                variant: 'caption',
                noWrap: true,
                color: activeId === h.id ? 'primary.main' : 'text.secondary',
                fontWeight: activeId === h.id ? 600 : 400,
              }}
            />
          </ListItemButton>
        ))}
      </List>
    </Box>
  );
}
