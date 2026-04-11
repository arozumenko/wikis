import { useState, useCallback } from 'react';
import {
  Alert,
  Box,
  Button,
  Checkbox,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControlLabel,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import PublicOutlinedIcon from '@mui/icons-material/PublicOutlined';
import { createProject, addWikiToProject, type ProjectResponse } from '../api/project';
import type { components } from '../api/types.generated';

type WikiSummary = components['schemas']['WikiSummary'];

interface CreateProjectDialogProps {
  open: boolean;
  onClose: () => void;
  onCreated: (p: ProjectResponse) => void;
  availableWikis: WikiSummary[];
}

function extractOwnerRepo(url: string): string {
  if (url.startsWith('/') || url.startsWith('file://')) {
    const raw = url.replace('file://', '');
    const base = raw.split('/').pop() ?? raw;
    const parts = base.split('_');
    if (parts.length >= 2) return `${parts[0]}/${parts[1]}`;
    return base;
  }
  try {
    const parts = new URL(url).pathname.split('/').filter(Boolean);
    if (parts.length >= 2) return `${parts[0]}/${parts[1]}`;
  } catch {
    /* not a URL */
  }
  return url;
}

export function CreateProjectDialog({
  open,
  onClose,
  onCreated,
  availableWikis,
}: CreateProjectDialogProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [visibility, setVisibility] = useState<'personal' | 'shared'>('personal');
  const [selectedWikiIds, setSelectedWikiIds] = useState<Set<string>>(new Set());
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const nameError =
    name.trim().length > 0 && (name.trim().length < 1 || name.trim().length > 100)
      ? 'Name must be between 1 and 100 characters'
      : null;

  const handleClose = useCallback(() => {
    if (submitting) return;
    setName('');
    setDescription('');
    setVisibility('personal');
    setSelectedWikiIds(new Set());
    setError(null);
    onClose();
  }, [submitting, onClose]);

  const handleToggleWiki = useCallback((wikiId: string) => {
    setSelectedWikiIds((prev) => {
      const next = new Set(prev);
      if (next.has(wikiId)) {
        next.delete(wikiId);
      } else {
        next.add(wikiId);
      }
      return next;
    });
  }, []);

  const handleSubmit = useCallback(async () => {
    const trimmedName = name.trim();
    if (!trimmedName || trimmedName.length > 100) return;
    setSubmitting(true);
    setError(null);
    try {
      const project = await createProject({
        name: trimmedName,
        description: description.trim() || undefined,
        visibility,
      });
      // Add selected wikis sequentially — failures are non-fatal
      for (const wikiId of selectedWikiIds) {
        try {
          await addWikiToProject(project.id, wikiId);
        } catch {
          // continue — partial success is acceptable
        }
      }
      onCreated(project);
      setName('');
      setDescription('');
      setVisibility('personal');
      setSelectedWikiIds(new Set());
    } catch {
      setError('Failed to create project. Please try again.');
    } finally {
      setSubmitting(false);
    }
  }, [name, description, visibility, selectedWikiIds, onCreated]);

  const canSubmit = name.trim().length >= 1 && name.trim().length <= 100 && !submitting;

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>New Project</DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 1 }}>
          {error && <Alert severity="error">{error}</Alert>}

          <TextField
            label="Project Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            fullWidth
            size="small"
            inputProps={{ maxLength: 100 }}
            error={!!nameError}
            helperText={nameError ?? `${name.trim().length}/100`}
            disabled={submitting}
          />

          <TextField
            label="Description (optional)"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            fullWidth
            size="small"
            multiline
            minRows={2}
            maxRows={4}
            disabled={submitting}
          />

          <Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Visibility
            </Typography>
            <ToggleButtonGroup
              value={visibility}
              exclusive
              onChange={(_e, val) => {
                if (val !== null) setVisibility(val as 'personal' | 'shared');
              }}
              size="small"
              disabled={submitting}
            >
              <ToggleButton value="personal">
                <LockOutlinedIcon fontSize="small" sx={{ mr: 0.75 }} />
                Personal
              </ToggleButton>
              <ToggleButton value="shared">
                <PublicOutlinedIcon fontSize="small" sx={{ mr: 0.75 }} />
                Shared
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>

          {availableWikis.length > 0 && (
            <>
              <Divider />
              <Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Add wikis to this project (optional)
                </Typography>
                <Box
                  sx={{
                    maxHeight: 200,
                    overflowY: 'auto',
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: 1,
                    px: 1,
                  }}
                >
                  {availableWikis.map((wiki) => (
                    <FormControlLabel
                      key={wiki.wiki_id}
                      control={
                        <Checkbox
                          size="small"
                          checked={selectedWikiIds.has(wiki.wiki_id)}
                          onChange={() => handleToggleWiki(wiki.wiki_id)}
                          disabled={submitting}
                        />
                      }
                      label={
                        <Box>
                          <Typography variant="body2">
                            {extractOwnerRepo(wiki.repo_url)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {wiki.repo_url}
                          </Typography>
                        </Box>
                      }
                      sx={{ display: 'flex', alignItems: 'flex-start', py: 0.5 }}
                    />
                  ))}
                </Box>
              </Box>
            </>
          )}
        </Box>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={handleClose} disabled={submitting}>
          Cancel
        </Button>
        <Button
          variant="contained"
          onClick={handleSubmit}
          disabled={!canSubmit}
          startIcon={submitting ? <CircularProgress size={16} /> : undefined}
        >
          {submitting ? 'Creating...' : 'Create Project'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
