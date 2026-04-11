import { useState } from 'react';
import {
  Button,
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

interface ExportButtonProps {
  wikiId: string;
  wikiTitle: string;
  isComplete: boolean;
}

export function ExportButton({ wikiId, isComplete }: ExportButtonProps) {
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  const open = Boolean(anchorEl);

  const handleOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleExport = async (format: 'obsidian' | 'wikis') => {
    handleClose();

    const url = `/api/v1/wikis/${encodeURIComponent(wikiId)}/export?format=${format}`;

    // Quick auth check — a HEAD request lets us detect 401 before navigating
    try {
      const resp = await fetch(url, { method: 'HEAD' });
      if (resp.status === 401 || resp.status === 403) {
        setSnackbarMessage('Session expired — please sign in again to export.');
        setSnackbarOpen(true);
        return;
      }
    } catch {
      // Network error — let the browser handle it on navigation
    }

    window.location.href = url;
  };

  const button = (
    <Button
      size="small"
      variant="outlined"
      startIcon={<FileDownloadIcon fontSize="small" />}
      onClick={isComplete ? handleOpen : undefined}
      disabled={!isComplete}
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
