import { useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  LinearProgress,
  Typography,
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { importWiki, type WikiSummary } from '../api/wiki';

interface ImportWikiDialogProps {
  open: boolean;
  onClose(): void;
  onSuccess(wiki: WikiSummary): void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isValidExtension(filename: string): boolean {
  return filename.endsWith('.wikiexport');
}

export function ImportWikiDialog({ open, onClose, onSuccess }: ImportWikiDialogProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const resetState = () => {
    setSelectedFile(null);
    setValidationError(null);
    setUploadError(null);
    setUploading(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleClose = () => {
    if (uploading) return;
    resetState();
    onClose();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    setUploadError(null);
    if (!file) {
      setSelectedFile(null);
      setValidationError(null);
      return;
    }
    if (!isValidExtension(file.name)) {
      setSelectedFile(null);
      setValidationError('Please select a .wikiexport bundle');
      return;
    }
    setValidationError(null);
    setSelectedFile(file);
  };

  const handleImport = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setUploadError(null);
    try {
      const wiki = await importWiki(selectedFile);
      resetState();
      onSuccess(wiki);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Import failed. Please try again.';
      setUploadError(message);
    } finally {
      setUploading(false);
    }
  };

  const canImport = selectedFile !== null && !validationError && !uploading;

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>Import Wiki</DialogTitle>
      <DialogContent>
        {uploading && <LinearProgress sx={{ mb: 2, borderRadius: 1 }} />}

        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Select a .wikiexport bundle to restore a wiki from a previous export.
        </Typography>

        <Box
          sx={{
            border: '2px dashed',
            borderColor: validationError ? 'error.main' : 'divider',
            borderRadius: 2,
            p: 3,
            textAlign: 'center',
            cursor: uploading ? 'default' : 'pointer',
            transition: 'border-color 0.2s ease',
            '&:hover': uploading
              ? {}
              : {
                  borderColor: 'primary.main',
                  bgcolor: 'action.hover',
                },
          }}
          onClick={() => {
            if (!uploading) fileInputRef.current?.click();
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".wikiexport"
            style={{ display: 'none' }}
            onChange={handleFileChange}
            disabled={uploading}
          />
          <UploadFileIcon sx={{ fontSize: 40, color: 'text.secondary', mb: 1 }} />
          {selectedFile ? (
            <Box>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                {selectedFile.name}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {formatBytes(selectedFile.size)}
              </Typography>
            </Box>
          ) : (
            <Typography variant="body2" color="text.secondary">
              Click to browse — .wikiexport
            </Typography>
          )}
        </Box>

        {validationError && (
          <Typography variant="caption" color="error" sx={{ mt: 1, display: 'block' }}>
            {validationError}
          </Typography>
        )}

        {uploadError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {uploadError}
          </Alert>
        )}
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={handleClose} disabled={uploading}>
          Cancel
        </Button>
        <Button
          variant="contained"
          onClick={handleImport}
          disabled={!canImport}
        >
          Import
        </Button>
      </DialogActions>
    </Dialog>
  );
}
