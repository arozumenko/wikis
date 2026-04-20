import { useCallback, useMemo, useState } from 'react';
import { Box, Collapse, IconButton, List, ListItemButton, ListItemText, Typography } from '@mui/material';

export interface WikiPage {
  id: string;
  title: string;
  order: number;
  section?: string;
}

interface WikiSidebarProps {
  pages: WikiPage[];
  activePageId: string | null;
  onSelectPage: (pageId: string) => void;
}

interface Section {
  name: string;
  indexPage?: WikiPage;
  pages: WikiPage[];
}

function groupBySection(pages: WikiPage[]): Section[] {
  const sorted = [...pages].sort((a, b) => a.order - b.order);
  const sections: Section[] = [];
  const unsectioned: WikiPage[] = [];

  for (const page of sorted) {
    const sectionName = page.section ?? inferSection(page.title);
    if (sectionName) {
      const existing = sections.find((s) => s.name === sectionName);
      if (existing) {
        // _index pages become the section header page, not a list item
        if (page.id.endsWith('/_index')) {
          existing.indexPage = page;
        } else {
          existing.pages.push(page);
        }
      } else {
        if (page.id.endsWith('/_index')) {
          sections.push({ name: sectionName, indexPage: page, pages: [] });
        } else {
          sections.push({ name: sectionName, pages: [page] });
        }
      }
    } else {
      unsectioned.push(page);
    }
  }

  if (unsectioned.length > 0) {
    sections.unshift({ name: '', pages: unsectioned });
  }

  return sections;
}

function inferSection(title: string): string {
  const colonIdx = title.indexOf(':');
  if (colonIdx > 0 && colonIdx < 20) {
    return title.slice(0, colonIdx).trim();
  }
  return '';
}

export function WikiSidebar({ pages, activePageId, onSelectPage }: WikiSidebarProps) {
  const sections = useMemo(() => groupBySection(pages), [pages]);

  const activeSectionName = useMemo(() => {
    for (const section of sections) {
      if (section.pages.some((p) => p.id === activePageId)) {
        return section.name;
      }
    }
    return null;
  }, [sections, activePageId]);

  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const toggleSection = useCallback((name: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);

  return (
    <Box
      component="nav"
      sx={{
        width: 240,
        flexShrink: 0,
        borderRight: '1px solid',
        borderColor: 'divider',
        height: '100%',
        overflow: 'auto',
        py: 2,
        scrollbarWidth: 'thin',
        scrollbarColor: 'rgba(255,255,255,0.15) transparent',
        '&::-webkit-scrollbar': { width: 6 },
        '&::-webkit-scrollbar-track': { background: 'transparent' },
        '&::-webkit-scrollbar-thumb': {
          background: 'rgba(255,255,255,0.15)',
          borderRadius: 3,
          '&:hover': { background: 'rgba(255,255,255,0.25)' },
        },
      }}
    >
      <List disablePadding sx={{ '& .MuiListItemButton-root': { borderRadius: 0 } }}>
        {sections.map((section) => {
          if (!section.name) {
            return section.pages.map((page) => (
              <PageItem
                key={page.id}
                page={page}
                active={page.id === activePageId}
                onClick={onSelectPage}
              />
            ));
          }

          const isOpen = !collapsed.has(section.name);
          const hasActivePage = section.name === activeSectionName;

          return (
            <Box key={section.name} sx={{ mb: 0.5 }}>
              <ListItemButton
                selected={hasActivePage && !section.pages.some((p) => p.id === activePageId)}
                onClick={() => {
                  if (section.indexPage) {
                    onSelectPage(section.indexPage.id);
                    // Ensure section is open when navigating to its index
                    setCollapsed((prev) => {
                      const next = new Set(prev);
                      next.delete(section.name);
                      return next;
                    });
                  } else {
                    toggleSection(section.name);
                  }
                }}
                sx={{
                  px: 2,
                  py: 0.5,
                  minHeight: 28,
                  pr: 1,
                  '&.Mui-selected': {
                    bgcolor: 'transparent',
                    borderLeft: '2px solid',
                    borderColor: 'primary.main',
                    '&:hover': { bgcolor: 'action.hover' },
                  },
                }}
              >
                <Typography
                  variant="caption"
                  sx={{
                    fontWeight: 600,
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    color: hasActivePage ? 'text.primary' : 'text.secondary',
                    fontSize: '0.68rem',
                    flex: 1,
                  }}
                >
                  {section.name}
                </Typography>
                <IconButton
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleSection(section.name);
                  }}
                  sx={{ p: 0.25, ml: 0.5, color: 'text.secondary' }}
                >
                  <Typography variant="caption" sx={{ fontSize: '0.7rem', lineHeight: 1 }}>
                    {isOpen ? '▾' : '▸'}
                  </Typography>
                </IconButton>
              </ListItemButton>
              <Collapse in={isOpen}>
                {section.pages.map((page) => (
                  <PageItem
                    key={page.id}
                    page={page}
                    active={page.id === activePageId}
                    onClick={onSelectPage}
                    indent
                  />
                ))}
              </Collapse>
            </Box>
          );
        })}
      </List>
    </Box>
  );
}

function PageItem({
  page,
  active,
  onClick,
  indent = false,
}: {
  page: WikiPage;
  active: boolean;
  onClick: (id: string) => void;
  indent?: boolean;
}) {
  return (
    <ListItemButton
      selected={active}
      onClick={() => onClick(page.id)}
      sx={{
        px: 2,
        pl: indent ? 3.5 : 2,
        py: 0.4,
        minHeight: 32,
        '&.Mui-selected': {
          bgcolor: 'transparent',
          borderLeft: '2px solid',
          borderColor: 'primary.main',
          '&:hover': { bgcolor: 'action.hover' },
        },
      }}
    >
      <ListItemText
        primary={page.title}
        primaryTypographyProps={{
          variant: 'body2',
          noWrap: true,
          fontSize: '0.82rem',
          fontWeight: active ? 600 : 400,
          color: active ? 'text.primary' : 'text.secondary',
        }}
      />
    </ListItemButton>
  );
}
