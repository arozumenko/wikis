import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  Autocomplete,
  Box,
  Button,
  Chip,
  CircularProgress,
  Grid,
  IconButton,
  Snackbar,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import EditOutlinedIcon from '@mui/icons-material/EditOutlined';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import PublicOutlinedIcon from '@mui/icons-material/PublicOutlined';
import RemoveCircleOutlineIcon from '@mui/icons-material/RemoveCircleOutline';
import { useNavigate, useParams } from 'react-router-dom';
import { ConfirmDialog } from '../components/ConfirmDialog';
import {
  getProject,
  updateProject,
  deleteProject,
  addWikiToProject,
  removeWikiFromProject,
  listProjectWikis,
  type ProjectResponse,
} from '../api/project';
import { listWikis } from '../api/wiki';
import { useAuth } from '../hooks/useAuth';
import type { components } from '../api/types.generated';

type WikiSummary = components['schemas']['WikiSummary'];

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

export function ProjectPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();

  const [project, setProject] = useState<ProjectResponse | null>(null);
  const [wikis, setWikis] = useState<WikiSummary[]>([]);
  const [allWikis, setAllWikis] = useState<WikiSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Inline edit state
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [savingEdit, setSavingEdit] = useState(false);

  // Delete project state
  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // Add wiki state
  const [wikiToAdd, setWikiToAdd] = useState<WikiSummary | null>(null);
  const [addingWiki, setAddingWiki] = useState(false);
  const [removingWikiId, setRemovingWikiId] = useState<string | null>(null);

  const [snackMessage, setSnackMessage] = useState<string | null>(null);
  const [snackSeverity, setSnackSeverity] = useState<'success' | 'error'>('error');

  const showSnack = useCallback((msg: string, severity: 'success' | 'error' = 'error') => {
    setSnackMessage(msg);
    setSnackSeverity(severity);
  }, []);

  const load = useCallback(async () => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    try {
      const [proj, wikiRes, allWikiRes] = await Promise.all([
        getProject(projectId),
        listProjectWikis(projectId),
        listWikis(),
      ]);
      setProject(proj);
      setWikis(wikiRes.wikis ?? []);
      setAllWikis(allWikiRes.wikis ?? []);
    } catch {
      setError('Failed to load project.');
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    load();
  }, [load]);

  const isOwner = user?.id === project?.owner_id;

  // Wikis not yet in this project
  const memberIds = new Set(wikis.map((w) => w.wiki_id));
  const availableToAdd = allWikis.filter((w) => !memberIds.has(w.wiki_id));

  const handleStartEdit = useCallback(() => {
    if (!project) return;
    setEditName(project.name);
    setEditDescription(project.description ?? '');
    setEditing(true);
  }, [project]);

  const handleCancelEdit = useCallback(() => {
    setEditing(false);
  }, []);

  const handleSaveEdit = useCallback(async () => {
    if (!project || !editName.trim()) return;
    setSavingEdit(true);
    try {
      const updated = await updateProject(project.id, {
        name: editName.trim(),
        description: editDescription.trim() || undefined,
      });
      setProject(updated);
      setEditing(false);
      showSnack('Project updated', 'success');
    } catch {
      showSnack('Failed to save changes');
    } finally {
      setSavingEdit(false);
    }
  }, [project, editName, editDescription, showSnack]);

  const handleDeleteProject = useCallback(async () => {
    if (!project) return;
    setDeleting(true);
    try {
      await deleteProject(project.id);
      navigate('/');
    } catch {
      showSnack('Failed to delete project');
      setDeleting(false);
    }
  }, [project, navigate, showSnack]);

  const handleAddWiki = useCallback(async () => {
    if (!project || !wikiToAdd) return;
    setAddingWiki(true);
    try {
      await addWikiToProject(project.id, wikiToAdd.wiki_id);
      setWikis((prev) => [...prev, wikiToAdd]);
      setProject((prev) => prev ? { ...prev, wiki_count: prev.wiki_count + 1 } : prev);
      setWikiToAdd(null);
      showSnack('Wiki added to project', 'success');
    } catch {
      showSnack('Failed to add wiki');
    } finally {
      setAddingWiki(false);
    }
  }, [project, wikiToAdd, showSnack]);

  const handleRemoveWiki = useCallback(
    async (wikiId: string) => {
      if (!project) return;
      setRemovingWikiId(wikiId);
      try {
        await removeWikiFromProject(project.id, wikiId);
        setWikis((prev) => prev.filter((w) => w.wiki_id !== wikiId));
        setProject((prev) => prev ? { ...prev, wiki_count: Math.max(0, prev.wiki_count - 1) } : prev);
        showSnack('Wiki removed from project', 'success');
      } catch {
        showSnack('Failed to remove wiki');
      } finally {
        setRemovingWikiId(null);
      }
    },
    [project, showSnack],
  );

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error || !project) {
    return (
      <Box sx={{ px: { xs: 3, sm: 4, md: 6 }, py: 4, maxWidth: 1000, mx: 'auto' }}>
        <Alert severity="error">{error ?? 'Project not found.'}</Alert>
        <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/')} sx={{ mt: 2 }}>
          Back to Dashboard
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ px: { xs: 3, sm: 4, md: 6, lg: 8 }, py: 4, maxWidth: 1000, mx: 'auto' }}>
      {/* Back */}
      <Button
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate('/')}
        sx={{ mb: 3, color: 'text.secondary' }}
        size="small"
      >
        All wikis
      </Button>

      {/* Header */}
      <Box sx={{ mb: 4 }}>
        {editing ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, maxWidth: 600 }}>
            <TextField
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              label="Project Name"
              size="small"
              fullWidth
              inputProps={{ maxLength: 100 }}
              disabled={savingEdit}
            />
            <TextField
              value={editDescription}
              onChange={(e) => setEditDescription(e.target.value)}
              label="Description"
              size="small"
              fullWidth
              multiline
              minRows={2}
              disabled={savingEdit}
            />
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                size="small"
                variant="contained"
                startIcon={savingEdit ? <CircularProgress size={14} /> : <CheckIcon />}
                onClick={handleSaveEdit}
                disabled={!editName.trim() || savingEdit}
              >
                Save
              </Button>
              <Button size="small" startIcon={<CloseIcon />} onClick={handleCancelEdit} disabled={savingEdit}>
                Cancel
              </Button>
            </Box>
          </Box>
        ) : (
          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
            <Box sx={{ flex: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                <Typography variant="h4" component="h1" sx={{ fontWeight: 700 }}>
                  {project.name}
                </Typography>
                <Chip
                  icon={
                    project.visibility === 'personal' ? (
                      <LockOutlinedIcon sx={{ fontSize: '0.8rem !important' }} />
                    ) : (
                      <PublicOutlinedIcon sx={{ fontSize: '0.8rem !important' }} />
                    )
                  }
                  label={project.visibility === 'personal' ? 'Personal' : 'Shared'}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: '0.7rem' }}
                />
              </Box>
              {project.description && (
                <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
                  {project.description}
                </Typography>
              )}
            </Box>
            {isOwner && (
              <Box sx={{ display: 'flex', gap: 0.5, flexShrink: 0 }}>
                <Tooltip title="Edit project">
                  <IconButton size="small" onClick={handleStartEdit}>
                    <EditOutlinedIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Delete project">
                  <IconButton
                    size="small"
                    disabled={deleting}
                    onClick={() => setConfirmDeleteOpen(true)}
                    sx={{ '&:hover': { color: 'error.main' } }}
                  >
                    <DeleteOutlineIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            )}
          </Box>
        )}
      </Box>

      {/* Wiki grid */}
      {wikis.length === 0 ? (
        <Alert
          severity="info"
          action={
            isOwner && availableToAdd.length > 0 ? (
              <Button color="inherit" size="small" onClick={() => {}}>
                Add wikis below
              </Button>
            ) : undefined
          }
        >
          No wikis in this project yet.
        </Alert>
      ) : (
        <Grid container spacing={2.5} sx={{ mb: 4 }}>
          {wikis.map((wiki) => {
            const isRemoving = removingWikiId === wiki.wiki_id;
            return (
              <Grid item xs={12} sm={6} md={4} key={wiki.wiki_id}>
                <Box
                  sx={{
                    position: 'relative',
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: 3,
                    px: 2.5,
                    py: 2,
                    bgcolor: 'background.paper',
                    cursor: 'pointer',
                    transition: 'box-shadow 0.2s ease',
                    '&:hover': { boxShadow: 3 },
                  }}
                  onClick={() => navigate(`/wiki/${wiki.wiki_id}`)}
                >
                  <Typography variant="body1" sx={{ fontWeight: 600 }} noWrap>
                    {extractOwnerRepo(wiki.repo_url)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {wiki.branch}
                    {wiki.page_count > 0 ? ` · ${wiki.page_count} pages` : ''}
                  </Typography>
                  {isOwner && (
                    <Box
                      sx={{ position: 'absolute', top: 6, right: 6 }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <Tooltip title="Remove from project">
                        <span>
                          <IconButton
                            size="small"
                            disabled={isRemoving}
                            onClick={() => handleRemoveWiki(wiki.wiki_id)}
                            sx={{
                              color: 'text.disabled',
                              '&:hover': { color: 'error.main' },
                            }}
                          >
                            {isRemoving ? (
                              <CircularProgress size={14} />
                            ) : (
                              <RemoveCircleOutlineIcon fontSize="small" />
                            )}
                          </IconButton>
                        </span>
                      </Tooltip>
                    </Box>
                  )}
                </Box>
              </Grid>
            );
          })}
        </Grid>
      )}

      {/* Add wiki panel — owner only */}
      {isOwner && availableToAdd.length > 0 && (
        <Box
          sx={{
            mt: 4,
            p: 2.5,
            border: '1px dashed',
            borderColor: 'divider',
            borderRadius: 3,
          }}
        >
          <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
            Add a wiki to this project
          </Typography>
          <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
            <Autocomplete
              options={availableToAdd}
              getOptionLabel={(opt) => extractOwnerRepo(opt.repo_url)}
              value={wikiToAdd}
              onChange={(_e, val) => setWikiToAdd(val)}
              size="small"
              sx={{ flex: 1 }}
              renderInput={(params) => (
                <TextField {...params} placeholder="Search wikis..." />
              )}
              isOptionEqualToValue={(a, b) => a.wiki_id === b.wiki_id}
            />
            <Button
              variant="contained"
              disabled={!wikiToAdd || addingWiki}
              onClick={handleAddWiki}
              startIcon={addingWiki ? <CircularProgress size={14} /> : undefined}
            >
              Add
            </Button>
          </Box>
        </Box>
      )}

      {/* Dialogs */}
      <ConfirmDialog
        open={confirmDeleteOpen}
        title="Delete Project"
        message="Delete this project? The wikis inside will not be deleted."
        onConfirm={handleDeleteProject}
        onCancel={() => setConfirmDeleteOpen(false)}
      />

      <Snackbar
        open={snackMessage !== null}
        autoHideDuration={4000}
        onClose={() => setSnackMessage(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setSnackMessage(null)}
          severity={snackSeverity}
          sx={{ width: '100%' }}
        >
          {snackMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
}
