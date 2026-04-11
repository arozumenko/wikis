import { useState } from 'react';
import {
  Button,
  CircularProgress,
  Menu,
  MenuItem,
  Tooltip,
  Snackbar,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import FolderZipIcon from '@mui/icons-material/FolderZip';
import InventoryIcon from '@mui/icons-material/Inventory';
import { getAuthToken } from '../api/client';

interface ExportButtonProps {
  wikiId: string;
  wikiTitle: string;
  isComplete: boolean;
}

/** Strip characters that are unsafe in filenames (keep alphanumeric, spaces, hyphens, underscores, dots). */
function sanitizeFilename(name: string): string {
  return name.replace(/[^\w\s.-]/g, '').trim() || 'export';
}

export function ExportButton({ wikiId, wikiTitle, isComplete }: ExportButtonProps) {
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [loading, setLoading] = useState(false);

  const open = Boolean(anchorEl);

  const handleOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const setError = (msg: string) => {
    setSnackbarMessage(msg);
    setSnackbarOpen(true);
  };

  const handleExport = async (format: 'obsidian' | 'wikis') => {
    setAnchorEl(null);
    setLoading(true);
    try {
      const token = await getAuthToken();
      const res = await fetch(`/api/v1/wikis/${encodeURIComponent(wikiId)}/export?format=${format}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) {
        if (res.status === 401 || res.status === 403) {
          setError('Session expired. Please log in again.');
        } else {
          setError(`Export failed (${res.status})`);
        }
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const base = sanitizeFilename(wikiTitle);
      const filename = format === 'wikis' ? `${base}.wikiexport` : `${base}.zip`;
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      setError('Export failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const isDisabled = !isComplete || loading;

  const button = (
    <Button
      size="small"
      variant="outlined"
      startIcon={loading ? <CircularProgress size={14} color="inherit" /> : <FileDownloadIcon fontSize="small" />}
      onClick={isDisabled ? undefined : handleOpen}
      disabled={isDisabled}
      aria-haspopup="true"
      aria-expanded={open}
      aria-controls={open ? 'export-menu' : undefined}
      sx={{
        textTransform: 'none',
        fontSize: '0.8rem',
        py: 0.4,
        px: 1.25,
        borderColor: 'divider',
        color: 'text.secondary',
        '&:hover': {
          borderColor: 'primary.main',
          color: 'primary.main',
        },
        '&.Mui-disabled': {
          borderColor: 'divider',
          color: 'text.disabled',
        },
      }}
    >
      Export
    </Button>
  );

  return (
    <>
      {isComplete ? (
        button
      ) : (
        <Tooltip title="Wiki must be fully generated before export">
          <span>{button}</span>
        </Tooltip>
      )}

      <Menu
        id="export-menu"
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        slotProps={{
          paper: {
            elevation: 3,
            sx: { minWidth: 220, mt: 0.5 },
          },
        }}
      >
        <MenuItem onClick={() => handleExport('obsidian')} dense>
          <ListItemIcon>
            <FolderZipIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="Obsidian Vault (.zip)"
            primaryTypographyProps={{ fontSize: '0.875rem' }}
          />
        </MenuItem>
        <MenuItem onClick={() => handleExport('wikis')} dense>
          <ListItemIcon>
            <InventoryIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="Wikis Bundle (.wikiexport)"
            primaryTypographyProps={{ fontSize: '0.875rem' }}
          />
        </MenuItem>
      </Menu>

      <Snackbar
        open={snackbarOpen}
        autoHideDuration={4000}
        onClose={() => setSnackbarOpen(false)}
        message={snackbarMessage}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      />
    </>
  );
}
