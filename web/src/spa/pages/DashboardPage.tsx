import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardActionArea,
  Chip,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  Grid,
  IconButton,
  InputAdornment,
  LinearProgress,
  Snackbar,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import AddIcon from '@mui/icons-material/Add';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import CreateNewFolderOutlinedIcon from '@mui/icons-material/CreateNewFolderOutlined';
import FileUploadIcon from '@mui/icons-material/FileUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import PublicOutlinedIcon from '@mui/icons-material/PublicOutlined';
import RefreshIcon from '@mui/icons-material/Refresh';
import ShareIcon from '@mui/icons-material/Share';
import { useNavigate } from 'react-router-dom';
import { ConfirmDialog } from '../components/ConfirmDialog';
import { GenerateForm } from '../components/GenerateForm';
import { ImportWikiDialog } from '../components/ImportWikiDialog';
import { ProjectCard } from '../components/ProjectCard';
import { CreateProjectDialog } from '../components/CreateProjectDialog';
import { listWikis, deleteWiki, generateWiki, updateWikiVisibility } from '../api/wiki';
import { listProjects, type ProjectResponse } from '../api/project';
import { ApiError } from '../api/client';
import type { components } from '../api/types.generated';

type WikiSummary = components['schemas']['WikiSummary'];
type VisibilityFilter = 'all' | 'mine' | 'shared' | 'projects';

const CARD_GRADIENTS = [
  'linear-gradient(135deg, #A855F7, #6366F1)', // violet → indigo
  'linear-gradient(135deg, #FACC15, #22C55E)', // yellow → green
  'linear-gradient(135deg, #E0C3FC, #F5A0E6)', // lavender → pink
  'linear-gradient(135deg, #22D3EE, #3B82F6)', // cyan → blue
  'linear-gradient(135deg, #F43F5E, #7C3AED)', // rose → purple
  'linear-gradient(135deg, #D9F99D, #34D399)', // lime → emerald
  'linear-gradient(135deg, #C084FC, #6366F1)', // purple → indigo
  'linear-gradient(135deg, #FB923C, #F43F5E)', // orange → rose
  'linear-gradient(135deg, #2DD4BF, #10B981)', // teal → emerald
  'linear-gradient(135deg, #F472B6, #A855F7)', // pink → violet
];

/** Stable gradient for a wiki — derived from wiki_id so it never changes. */
function cardGradient(wikiId: string): string {
  let hash = 0;
  for (let i = 0; i < wikiId.length; i++) {
    hash = (hash * 31 + wikiId.charCodeAt(i)) | 0;
  }
  return CARD_GRADIENTS[Math.abs(hash) % CARD_GRADIENTS.length];
}

function extractOwnerRepo(url: string): string {
  // Handle local filesystem paths (e.g. /app/data/cache/owner_repo_branch_hash)
  if (url.startsWith('/') || url.startsWith('file://')) {
    const raw = url.replace('file://', '');
    const base = raw.split('/').pop() ?? raw;
    // Convention: owner_repo_branch_hash → owner/repo
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

export function DashboardPage() {
  const navigate = useNavigate();
  const [wikis, setWikis] = useState<WikiSummary[]>([]);
  const [projects, setProjects] = useState<ProjectResponse[]>([]);
  const [loadingProjects, setLoadingProjects] = useState(false);
  const [loading, setLoading] = useState(true);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [searchUrl, setSearchUrl] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [showGenerateModal, setShowGenerateModal] = useState(false);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [showCreateProjectDialog, setShowCreateProjectDialog] = useState(false);
  const [visibilityFilter, setVisibilityFilter] = useState<VisibilityFilter>('all');
  const [togglingVisibility, setTogglingVisibility] = useState<string | null>(null);
  const [snackbarMessage, setSnackbarMessage] = useState<string | null>(null);
  const [snackbarSeverity, setSnackbarSeverity] = useState<'success' | 'error'>('error');

  const showSnackbar = useCallback((message: string, severity: 'success' | 'error' = 'error') => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
  }, []);

  const fetchWikis = useCallback(() => {
    listWikis()
      .then((data) => setWikis(data.wikis ?? []))
      .catch(() => setWikis([]))
      .finally(() => setLoading(false));
  }, []);

  const fetchProjects = useCallback(() => {
    setLoadingProjects(true);
    listProjects()
      .then((data) => setProjects(data.projects ?? []))
      .catch(() => setProjects([]))
      .finally(() => setLoadingProjects(false));
  }, []);

  useEffect(() => {
    fetchWikis();
  }, [fetchWikis]);

  useEffect(() => {
    if (visibilityFilter === 'projects') {
      fetchProjects();
    }
  }, [visibilityFilter, fetchProjects]);

  // Auto-refresh every 10s while any wiki is generating
  useEffect(() => {
    const hasGenerating = wikis.some((w) => w.status === 'generating');
    if (!hasGenerating) return;
    const interval = setInterval(fetchWikis, 10000);
    return () => clearInterval(interval);
  }, [wikis, fetchWikis]);

  // Filter wikis by search query
  const query = searchUrl.trim().toLowerCase();
  const searchFilteredWikis = query
    ? wikis.filter((w) => {
        const label = extractOwnerRepo(w.repo_url).toLowerCase();
        const url = w.repo_url.toLowerCase();
        return label.includes(query) || url.includes(query);
      })
    : wikis;

  // Apply visibility filter on top of search
  const filteredWikis = searchFilteredWikis.filter((w) => {
    if (visibilityFilter === 'mine') return w.is_owner ?? true;
    if (visibilityFilter === 'shared') return (w.visibility ?? 'personal') === 'shared';
    return true;
  });

  const hasSearchWithNoResults = query.length > 0 && filteredWikis.length === 0;

  const handleView = useCallback(
    (wikiId: string) => {
      navigate(`/wiki/${wikiId}`);
    },
    [navigate],
  );

  const handleDelete = useCallback(
    (wiki: WikiSummary) => {
      const isOwner = wiki.is_owner ?? false;
      if (!isOwner) {
        showSnackbar('Only the wiki owner can perform this action');
        return;
      }
      setDeleteTarget(wiki.wiki_id);
    },
    [showSnackbar],
  );

  const confirmDelete = useCallback(async () => {
    if (!deleteTarget) return;
    try {
      await deleteWiki(deleteTarget);
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) {
        showSnackbar('Only the wiki owner can perform this action');
        setDeleteTarget(null);
        return;
      }
      /* ignore — invocation-only wikis return 404 but are purged server-side */
    }
    setWikis((prev) => prev.filter((w) => w.wiki_id !== deleteTarget));
    setDeleteTarget(null);
  }, [deleteTarget, showSnackbar]);

  const handleToggleVisibility = useCallback(
    async (wiki: WikiSummary) => {
      const newVisibility = (wiki.visibility ?? 'personal') === 'personal' ? 'shared' : 'personal';
      setTogglingVisibility(wiki.wiki_id);
      try {
        await updateWikiVisibility(wiki.wiki_id, newVisibility);
        setWikis((prev) =>
          prev.map((w) => (w.wiki_id === wiki.wiki_id ? { ...w, visibility: newVisibility } : w)),
        );
        showSnackbar(
          newVisibility === 'shared' ? 'Wiki is now shared with all users' : 'Wiki is now private',
          'success',
        );
      } catch (err) {
        if (err instanceof ApiError && err.status === 403) {
          showSnackbar('Only the wiki owner can change visibility');
        } else {
          showSnackbar('Failed to update visibility. Please try again.');
        }
      } finally {
        setTogglingVisibility(null);
      }
    },
    [showSnackbar],
  );

  const handleRefresh = useCallback(
    async (e: React.MouseEvent, wiki: WikiSummary) => {
      e.stopPropagation();
      const isOwner = wiki.is_owner ?? false;
      if (!isOwner) {
        showSnackbar('Only the wiki owner can perform this action');
        return;
      }
      // Refresh logic — navigate to the wiki viewer which handles refresh
      navigate(`/wiki/${wiki.wiki_id}`);
    },
    [navigate, showSnackbar],
  );

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ px: { xs: 3, sm: 4, md: 6, lg: 8 }, py: 4, maxWidth: 1000, mx: 'auto' }}>
      <Box sx={{ textAlign: 'center', mb: 5, mt: 4 }}>
        <Typography
          variant="h3"
          component="h1"
          sx={{
            fontFamily: '"Playfair Display", Georgia, serif',
            fontWeight: 700,
            mb: 2,
            fontSize: { xs: '1.8rem', md: '2.5rem' },
          }}
        >
          Which repo would you like to understand?
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, maxWidth: 750, mx: 'auto', mt: 3 }}>
          <TextField
            value={searchUrl}
            onChange={(e) => setSearchUrl(e.target.value)}
            placeholder="Search wikis or paste a URL to generate..."
            fullWidth
            size="small"
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon color="action" />
                </InputAdornment>
              ),
              sx: {
                borderRadius: 3,
                fontSize: '0.95rem',
                transition: 'box-shadow 0.3s ease',
                '&.Mui-focused': {
                  boxShadow:
                    '0 0 15px rgba(168, 85, 247, 0.4), 0 0 30px rgba(236, 72, 153, 0.2), 0 0 45px rgba(59, 130, 246, 0.15)',
                },
                '& .MuiOutlinedInput-notchedOutline': {
                  transition: 'border-color 0.3s ease',
                },
                '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                  borderColor: '#a855f7',
                  borderWidth: 1,
                },
              },
            }}
          />
          <ToggleButtonGroup
            value={visibilityFilter}
            exclusive
            onChange={(_e, val) => {
              if (val !== null) setVisibilityFilter(val as VisibilityFilter);
            }}
            size="small"
            sx={{
              flexShrink: 0,
              '& .MuiToggleButton-root': {
                px: 2,
                py: 0.85,
                fontSize: '0.8rem',
                textTransform: 'none',
                borderRadius: '20px !important',
                border: '1px solid',
                borderColor: 'divider',
                mx: 0.5,
                whiteSpace: 'nowrap',
                transition: 'all 0.3s ease',
                '&:hover': {
                  boxShadow:
                    '0 0 12px rgba(168, 85, 247, 0.35), 0 0 24px rgba(236, 72, 153, 0.2), 0 0 36px rgba(59, 130, 246, 0.12)',
                  borderColor: '#a855f7',
                },
                '&.Mui-selected': {
                  bgcolor: 'primary.main',
                  color: 'primary.contrastText',
                  borderColor: 'primary.main',
                  boxShadow:
                    '0 0 14px rgba(255, 107, 74, 0.4), 0 0 28px rgba(255, 107, 74, 0.2)',
                  '&:hover': {
                    bgcolor: 'primary.dark',
                    boxShadow:
                      '0 0 20px rgba(255, 107, 74, 0.5), 0 0 40px rgba(255, 107, 74, 0.25), 0 0 60px rgba(255, 107, 74, 0.15)',
                  },
                },
              },
              '& .MuiToggleButtonGroup-grouped': {
                borderRadius: '20px !important',
                '&:not(:first-of-type)': {
                  borderLeft: '1px solid',
                  borderColor: 'divider',
                  marginLeft: '4px',
                },
              },
            }}
          >
            <ToggleButton value="all">All</ToggleButton>
            <ToggleButton value="mine">My Wikis</ToggleButton>
            <ToggleButton value="shared">Shared</ToggleButton>
            <ToggleButton value="projects">Projects</ToggleButton>
          </ToggleButtonGroup>
          {visibilityFilter === 'projects' && (
            <Button
              variant="contained"
              size="small"
              startIcon={<CreateNewFolderOutlinedIcon />}
              onClick={() => setShowCreateProjectDialog(true)}
              sx={{ flexShrink: 0, whiteSpace: 'nowrap', borderRadius: 3 }}
            >
              New Project
            </Button>
          )}
          {visibilityFilter !== 'projects' && (
            <Button
              variant="outlined"
              size="small"
              startIcon={<FileUploadIcon />}
              onClick={() => setShowImportDialog(true)}
              sx={{ flexShrink: 0, whiteSpace: 'nowrap', borderRadius: 3 }}
            >
              Import Wiki
            </Button>
          )}
        </Box>
      </Box>

      {hasSearchWithNoResults && (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
            No wikis found for "{searchUrl.trim()}"
          </Typography>
          <Card
            variant="outlined"
            sx={{
              maxWidth: 320,
              mx: 'auto',
              borderRadius: 3,
              border: '2px dashed',
              borderColor: 'primary.main',
              cursor: 'pointer',
              '&:hover': { bgcolor: 'action.hover' },
            }}
            onClick={() => setShowGenerateModal(true)}
          >
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 1,
                py: 2.5,
                px: 3,
              }}
            >
              <AddCircleOutlineIcon color="primary" />
              <Typography variant="body2" color="primary" sx={{ fontWeight: 600 }}>
                Generate a wiki for this repo
              </Typography>
            </Box>
          </Card>
        </Box>
      )}

      {visibilityFilter === 'projects' ? (
        <Grid container spacing={2.5}>
          {loadingProjects ? (
            <Grid item xs={12}>
              <Box display="flex" justifyContent="center" py={4}>
                <CircularProgress />
              </Box>
            </Grid>
          ) : projects.length === 0 ? (
            <Grid item xs={12}>
              <Box sx={{ textAlign: 'center', py: 6 }}>
                <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
                  No projects yet. Create one to group your wikis.
                </Typography>
                <Button
                  variant="outlined"
                  startIcon={<CreateNewFolderOutlinedIcon />}
                  onClick={() => setShowCreateProjectDialog(true)}
                >
                  Create your first project
                </Button>
              </Box>
            </Grid>
          ) : (
            projects.map((project) => (
              <Grid item xs={12} sm={6} md={4} key={project.id}>
                <ProjectCard
                  project={project}
                  onDelete={() => setProjects((prev) => prev.filter((p) => p.id !== project.id))}
                />
              </Grid>
            ))
          )}
        </Grid>
      ) : null}

      {visibilityFilter !== 'projects' && <Grid container spacing={2.5}>
        {!hasSearchWithNoResults && (
          <Grid item xs={12} sm={6} md={4}>
            <Card
              variant="outlined"
              sx={{
                height: 140,
                borderRadius: 3,
                border: '2px dashed',
                borderColor: 'divider',
                '&:hover': { borderColor: 'primary.main' },
              }}
            >
              <CardActionArea
                onClick={() => setShowGenerateModal(true)}
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                }}
              >
                <AddIcon sx={{ fontSize: 32, color: 'text.secondary', mb: 1 }} />
                <Typography variant="body2" color="text.secondary">
                  Add a repository
                </Typography>
              </CardActionArea>
            </Card>
          </Grid>
        )}
        {filteredWikis.map((wiki) => {
          const isGenerating = wiki.status === 'generating';
          const isFailed = wiki.status === 'failed';
          const isOwner = wiki.is_owner ?? false;
          const visibility = wiki.visibility ?? 'personal';
          const isTogglingThis = togglingVisibility === wiki.wiki_id;
          const gradient = isFailed
            ? 'linear-gradient(135deg, #EF4444, #DC2626)'
            : cardGradient(wiki.wiki_id);

          return (
            <Grid item xs={12} sm={6} md={4} key={wiki.wiki_id}>
              {/* Wrapper for gradient glow pseudo-element */}
              <Box
                sx={{
                  position: 'relative',
                  borderRadius: 3,
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    inset: 0,
                    borderRadius: 'inherit',
                    background: gradient,
                    opacity: 0,
                    filter: 'blur(18px)',
                    transition: 'opacity 0.3s ease',
                    zIndex: 0,
                  },
                  '&:hover::before': {
                    opacity: 0.45,
                  },
                  '&:hover > .MuiCard-root': {
                    transform: 'translateY(-2px)',
                  },
                }}
              >
              <Card
                elevation={0}
                sx={{
                  position: 'relative',
                  zIndex: 1,
                  height: 140,
                  borderRadius: 3,
                  bgcolor: 'background.paper',
                  border: '1px solid',
                  borderColor: isFailed ? 'error.main' : 'divider',
                  transition: 'transform 0.2s ease',
                }}
              >
                  <Box
                    onClick={() => {
                      if (isGenerating && wiki.invocation_id) {
                        navigate(
                          `/wiki/${wiki.wiki_id}?generating=true&invocation=${wiki.invocation_id}`,
                        );
                      } else {
                        handleView(wiki.wiki_id);
                      }
                    }}
                    sx={{
                      height: '100%',
                      px: 2.5,
                      py: 2,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'flex-start',
                      justifyContent: 'space-between',
                      cursor: 'pointer',
                    }}
                  >
                    <Box sx={{ width: '100%' }}>
                      <Typography
                        variant="body1"
                        sx={{ fontWeight: 600, mb: 0.5 }}
                        noWrap
                      >
                        {extractOwnerRepo(wiki.repo_url)}
                      </Typography>
                      {isGenerating ? (
                        <Box sx={{ width: '100%' }}>
                          <LinearProgress
                            variant="determinate"
                            value={(wiki.progress ?? 0) * 100}
                            sx={{ height: 4, borderRadius: 2, mb: 0.5 }}
                          />
                          <Typography variant="caption" color="text.secondary">
                            {Math.round((wiki.progress ?? 0) * 100)}%
                          </Typography>
                        </Box>
                      ) : isFailed ? (
                        <Tooltip title={wiki.error ?? 'Generation failed'} arrow>
                          <Typography
                            variant="caption"
                            color="error.main"
                            noWrap
                            sx={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis' }}
                          >
                            {wiki.error ?? 'Generation failed'}
                          </Typography>
                        </Tooltip>
                      ) : (
                        <Typography variant="caption" color="text.secondary">
                          {wiki.branch}
                          {wiki.page_count > 0 ? ` · ${wiki.page_count} pages` : ''}
                        </Typography>
                      )}
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mt: 0.5 }}>
                        {isGenerating && (
                          <Chip
                            label="Generating"
                            size="small"
                            color="info"
                            sx={{ height: 18, fontSize: '0.6rem' }}
                          />
                        )}
                        {isFailed && (
                          <Chip
                            icon={<ErrorOutlineIcon />}
                            label="Failed"
                            size="small"
                            color="error"
                            sx={{ height: 18, fontSize: '0.6rem' }}
                          />
                        )}
                        <Chip
                          icon={
                            visibility === 'personal' ? (
                              <LockOutlinedIcon sx={{ fontSize: '0.75rem !important' }} />
                            ) : (
                              <PublicOutlinedIcon sx={{ fontSize: '0.75rem !important' }} />
                            )
                          }
                          label={visibility === 'personal' ? 'Personal' : 'Shared'}
                          size="small"
                          variant="outlined"
                          sx={{
                            height: 18,
                            fontSize: '0.6rem',
                            opacity: 0.7,
                            '& .MuiChip-label': { px: 0.75 },
                            '& .MuiChip-icon': { ml: 0.5 },
                          }}
                        />
                      </Box>
                    </Box>
                    <Box
                      sx={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        width: '100%',
                      }}
                    >
                      <Typography variant="caption" color="text.secondary">
                        {new Date(wiki.created_at).toLocaleDateString()}
                      </Typography>
                      <Box onClick={(e) => e.stopPropagation()}>
                        {/* Share/unshare toggle — owner only, not for failed wikis */}
                        {isOwner && !isFailed && (
                          <Tooltip
                            title={
                              visibility === 'personal'
                                ? 'Share with all users'
                                : 'Make private'
                            }
                          >
                            <span>
                              <IconButton
                                size="small"
                                disabled={isTogglingThis}
                                onClick={() => handleToggleVisibility(wiki)}
                                sx={{
                                  color:
                                    visibility === 'shared' ? 'primary.main' : 'text.secondary',
                                }}
                              >
                                <ShareIcon fontSize="small" />
                              </IconButton>
                            </span>
                          </Tooltip>
                        )}
                        {/* Refresh button — failed wikis, owner only */}
                        {isFailed && (
                          <Tooltip
                            title={
                              isOwner ? 'Retry generation' : 'Only the wiki owner can refresh'
                            }
                          >
                            <span>
                              <IconButton
                                size="small"
                                disabled={!isOwner}
                                onClick={(e) => handleRefresh(e, wiki)}
                                sx={{ color: 'text.secondary' }}
                              >
                                <RefreshIcon fontSize="small" />
                              </IconButton>
                            </span>
                          </Tooltip>
                        )}
                        {/* Delete button */}
                        <Tooltip
                          title={
                            isOwner ? 'Delete wiki' : 'Only the wiki owner can delete'
                          }
                        >
                          <span>
                            <IconButton
                              size="small"
                              disabled={!isOwner}
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDelete(wiki);
                              }}
                              sx={{
                                color: 'text.secondary',
                                '&:hover': { color: 'error.main' },
                                '&.Mui-disabled': { color: 'text.disabled' },
                              }}
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </span>
                        </Tooltip>
                      </Box>
                    </Box>
                  </Box>
                </Card>
              </Box>
            </Grid>
          );
        })}
      </Grid>}

      <Dialog
        open={showGenerateModal}
        onClose={() => setShowGenerateModal(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Generate Wiki</DialogTitle>
        <DialogContent>
          <GenerateForm
            key={searchUrl.trim()}
            disabled={submitting}
            initialUrl={searchUrl.trim()}
            onSubmit={async (req) => {
              setSubmitting(true);
              try {
                const response = await generateWiki(req);
                setShowGenerateModal(false);
                navigate(
                  `/wiki/${response.wiki_id}?generating=true&invocation=${response.invocation_id}`,
                );
              } catch (err: unknown) {
                // 409 Conflict — wiki already exists, navigate to it
                if (err && typeof err === 'object' && 'status' in err && (err as { status: number }).status === 409) {
                  const body = (err as { body?: { detail?: { wiki_id?: string } } }).body;
                  const wikiId = body?.detail?.wiki_id;
                  setShowGenerateModal(false);
                  if (wikiId) {
                    navigate(`/wiki/${wikiId}`);
                  }
                  // If no wiki_id, just close the modal — the existing wiki is visible on the dashboard
                  return;
                }
                throw err;
              } finally {
                setSubmitting(false);
              }
            }}
          />
        </DialogContent>
      </Dialog>

      <ImportWikiDialog
        open={showImportDialog}
        onClose={() => setShowImportDialog(false)}
        onSuccess={(wiki) => {
          setWikis((prev) => [wiki, ...prev]);
          setShowImportDialog(false);
          showSnackbar('Wiki imported successfully', 'success');
          navigate(`/wiki/${wiki.wiki_id}`);
        }}
      />

      <ConfirmDialog
        open={deleteTarget !== null}
        title="Delete Wiki"
        message="Are you sure? This cannot be undone."
        onConfirm={confirmDelete}
        onCancel={() => setDeleteTarget(null)}
      />

      <CreateProjectDialog
        open={showCreateProjectDialog}
        onClose={() => setShowCreateProjectDialog(false)}
        availableWikis={wikis}
        onCreated={(project) => {
          setProjects((prev) => [project, ...prev]);
          setShowCreateProjectDialog(false);
        }}
      />

      <Snackbar
        open={snackbarMessage !== null}
        autoHideDuration={4000}
        onClose={() => setSnackbarMessage(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setSnackbarMessage(null)}
          severity={snackbarSeverity}
          sx={{ width: '100%' }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
}
