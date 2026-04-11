import { useEffect, useRef, useState } from 'react';
import {
  Box,
  Chip,
  CircularProgress,
  Dialog,
  DialogContent,
  List,
  ListItemButton,
  ListItemText,
  TextField,
  Typography,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { searchWiki, searchProject } from '../api/search';
import type { SearchResultItem } from '../api/search';

export interface QuickSearchProps {
  open: boolean;
  onClose: () => void;
  wikiId?: string;
  projectId?: string;
}

export function QuickSearch({ open, onClose, wikiId, projectId }: QuickSearchProps) {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Reset state when dialog opens/closes
  useEffect(() => {
    if (!open) {
      setQuery('');
      setResults([]);
      setLoading(false);
      setSelectedIndex(0);
    }
  }, [open]);

  // Debounced search — 300ms after the last keystroke
  useEffect(() => {
    if (query.length < 2) {
      setResults([]);
      setSelectedIndex(0);
      return;
    }

    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        let response;
        if (wikiId) {
          response = await searchWiki(wikiId, query);
        } else if (projectId) {
          response = await searchProject(projectId, query);
        } else {
          return;
        }
        setResults(response.results);
        setSelectedIndex(0);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [query, wikiId, projectId]);

  // Reset selectedIndex whenever results change
  useEffect(() => {
    setSelectedIndex(0);
  }, [results]);

  function handleSelect(result: SearchResultItem) {
    const targetWikiId = result.wiki_id;
    // WikiViewerPage matches by ?page_title= (title-based) or ?page= (id-based).
    // Search results carry page_title but not the internal page id, so we use page_title.
    navigate(`/wiki/${targetWikiId}?page_title=${encodeURIComponent(result.page_title)}`);
    onClose();
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => (results.length > 0 ? Math.min(prev + 1, results.length - 1) : 0));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter') {
      const selected = results[selectedIndex];
      if (selected) {
        handleSelect(selected);
      }
    }
  }

  const hasContext = Boolean(wikiId || projectId);
  const showEmptyState = hasContext && query.length >= 2 && !loading && results.length === 0;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: { mt: '10vh', verticalAlign: 'top' },
      }}
    >
      <DialogContent sx={{ p: 0 }}>
        <Box sx={{ px: 2, pt: 2, pb: 1 }}>
          <TextField
            inputRef={inputRef}
            autoFocus
            fullWidth
            placeholder={hasContext ? 'Search pages…' : 'Open a wiki or project to search'}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!hasContext}
            variant="outlined"
            size="small"
            InputProps={{
              endAdornment: loading ? <CircularProgress size={18} /> : null,
            }}
          />
        </Box>

        {!hasContext && (
          <Box sx={{ px: 2, pb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Open a wiki or project to search
            </Typography>
          </Box>
        )}

        {showEmptyState && (
          <Box sx={{ px: 2, pb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              No results found
            </Typography>
          </Box>
        )}

        {results.length > 0 && (
          <List dense disablePadding sx={{ pb: 1 }}>
            {results.map((result, index) => (
              <ListItemButton
                key={`${result.wiki_id}::${result.page_title}`}
                selected={index === selectedIndex}
                onClick={() => handleSelect(result)}
                sx={{ px: 2, py: 1 }}
              >
                <ListItemText
                  primary={result.page_title}
                  secondary={
                    <Box component="span" sx={{ display: 'block' }}>
                      <Box
                        component="span"
                        sx={{
                          display: 'block',
                          fontSize: '0.78rem',
                          color: 'text.secondary',
                          mb: result.neighbors.length > 0 ? 0.5 : 0,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          maxWidth: '100%',
                        }}
                      >
                        {result.snippet.length > 120
                          ? `${result.snippet.slice(0, 120)}…`
                          : result.snippet}
                      </Box>
                      {result.neighbors.length > 0 && (
                        <Box component="span" sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {result.neighbors.slice(0, 3).map((neighbor) => (
                            <Chip
                              key={neighbor.title}
                              label={neighbor.title}
                              size="small"
                              variant="outlined"
                              sx={{ fontSize: '0.7rem', height: 20 }}
                            />
                          ))}
                        </Box>
                      )}
                    </Box>
                  }
                  secondaryTypographyProps={{ component: 'span' }}
                />
                <Typography
                  variant="caption"
                  color="text.disabled"
                  sx={{ ml: 1, flexShrink: 0, alignSelf: 'flex-start', pt: 0.5 }}
                >
                  {result.score.toFixed(2)}
                </Typography>
              </ListItemButton>
            ))}
          </List>
        )}
      </DialogContent>
    </Dialog>
  );
}
