import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Alert,
  Autocomplete,
  Box,
  Button,
  Chip,
  CircularProgress,
  Divider,
  Grid,
  IconButton,
  InputAdornment,
  Snackbar,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import BoltIcon from '@mui/icons-material/Bolt';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import EditOutlinedIcon from '@mui/icons-material/EditOutlined';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import PublicOutlinedIcon from '@mui/icons-material/PublicOutlined';
import QuestionAnswerOutlinedIcon from '@mui/icons-material/QuestionAnswerOutlined';
import RemoveCircleOutlineIcon from '@mui/icons-material/RemoveCircleOutline';
import SearchIcon from '@mui/icons-material/Search';
import SendIcon from '@mui/icons-material/Send';
import { useNavigate, useParams } from 'react-router-dom';
import { ConfirmDialog } from '../components/ConfirmDialog';
import { AnswerView } from '../components/AnswerView';
import { ToolCallPanel } from '../components/ToolCallPanel';
import {
  getProject,
  updateProject,
  deleteProject,
  addWikiToProject,
  removeWikiFromProject,
  listProjectWikis,
  askProject,
  researchProject,
  type ProjectResponse,
} from '../api/project';
import { listWikis } from '../api/wiki';
import { useAuth } from '../hooks/useAuth';
import type { components } from '../api/types.generated';
import type { ToolCallRecord, TodoItem } from '../api/sse';

type WikiSummary = components['schemas']['WikiSummary'];
type SourceReference = components['schemas']['SourceReference'];

type QAMode = 'fast' | 'deep';

interface QATurn {
  question: string;
  answer: string | null;
  sources: SourceReference[];
  toolCalls: ToolCallRecord[];
  todos: TodoItem[];
  loading: boolean;
  error: boolean;
  mode: QAMode;
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

  // Q&A panel state
  const [qaInput, setQaInput] = useState('');
  const [qaMode, setQaMode] = useState<QAMode>('fast');
  const [qaTurns, setQaTurns] = useState<QATurn[]>([]);
  const cancelQaRef = useRef<(() => void) | null>(null);

  // Cancel any in-flight SSE stream on unmount to prevent memory leaks.
  useEffect(() => () => { cancelQaRef.current?.(); }, []);

  const showSnack = useCallback((msg: string, severity: 'success' | 'error' = 'error') => {
    setSnackMessage(msg);
    setSnackSeverity(severity);
  }, []);

  const load = useCallback(async () => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    try {
      const [proj, wikiRes] = await Promise.all([
        getProject(projectId),
        listProjectWikis(projectId),
      ]);
      setProject(proj);
      setWikis(wikiRes.wikis ?? []);
      // Only fetch all wikis if the current user is the owner (needed for the "add wiki" panel)
      if (user?.id === proj.owner_id) {
        const allWikiRes = await listWikis();
        setAllWikis(allWikiRes.wikis ?? []);
      }
    } catch {
      setError('Failed to load project.');
    } finally {
      setLoading(false);
    }
  }, [projectId, user?.id]);

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

  const updateLastQaTurn = useCallback((updater: (prev: QATurn) => QATurn) => {
    setQaTurns((prev) => {
      if (!prev.length) return prev;
      return [...prev.slice(0, -1), updater(prev[prev.length - 1])];
    });
  }, []);

  const handleQaSubmit = useCallback(() => {
    const question = qaInput.trim();
    if (!question || !projectId) return;

    cancelQaRef.current?.();
    cancelQaRef.current = null;

    setQaTurns((prev) => [
      ...prev,
      { question, answer: null, sources: [], toolCalls: [], todos: [], loading: true, error: false, mode: qaMode },
    ]);
    setQaInput('');

    if (qaMode === 'deep') {
      const cancel = researchProject(
        projectId,
        question,
        (event) => {
          if (event.type === 'thinking_step') {
            const e = event;
            const stepKind = e.step_type ?? e.stepType;
            updateLastQaTurn((prev) => {
              if (stepKind === 'tool_call') {
                const record: ToolCallRecord = {
                  tool_name: e.tool,
                  tool_input: e.input ?? '',
                  tool_output: null,
                  timestamp: e.timestamp,
                  endTimestamp: null,
                  done: false,
                };
                return { ...prev, toolCalls: [...prev.toolCalls, record] };
              }
              if (stepKind === 'tool_result') {
                let updated = false;
                const toolCalls = [...prev.toolCalls]
                  .reverse()
                  .map((tc) => {
                    if (!updated && !tc.done && tc.tool_name === e.tool) {
                      updated = true;
                      return {
                        ...tc,
                        tool_output: e.output ?? e.output_preview ?? e.outputPreview ?? '',
                        endTimestamp: e.timestamp,
                        done: true,
                      };
                    }
                    return tc;
                  })
                  .reverse();
                return { ...prev, toolCalls };
              }
              return prev;
            });
          } else if (event.type === 'answer_chunk') {
            updateLastQaTurn((prev) => ({ ...prev, answer: (prev.answer ?? '') + event.chunk }));
          } else if (event.type === 'research_complete') {
            updateLastQaTurn((prev) => ({ ...prev, answer: event.report ?? '', loading: false, error: false }));
          } else if (event.type === 'research_error') {
            updateLastQaTurn((prev) => ({ ...prev, answer: `Error: ${event.error}`, loading: false, error: true }));
          } else if (event.type === 'task_failed') {
            updateLastQaTurn((prev) => ({ ...prev, answer: `Error: ${event.error}`, loading: false, error: true }));
          } else if (event.type === 'todo_update') {
            const todos = (event.todos as TodoItem[]) ?? [];
            updateLastQaTurn((prev) => ({ ...prev, todos }));
          }
        },
        () => {
          updateLastQaTurn((prev) => (prev.loading ? { ...prev, loading: false } : prev));
        },
        () => {
          updateLastQaTurn((prev) => ({
            ...prev,
            answer: 'Sorry, something went wrong. Please try again.',
            loading: false,
            error: true,
          }));
        },
      );
      cancelQaRef.current = cancel;
    } else {
      const cancel = askProject(
        projectId,
        question,
        (event) => {
          if (event.type === 'thinking_step') {
            const e = event;
            const stepKind = e.step_type ?? e.stepType;
            updateLastQaTurn((prev) => {
              if (stepKind === 'tool_call') {
                const record: ToolCallRecord = {
                  tool_name: e.tool,
                  tool_input: e.input ?? '',
                  tool_output: null,
                  timestamp: e.timestamp,
                  endTimestamp: null,
                  done: false,
                };
                return { ...prev, toolCalls: [...prev.toolCalls, record] };
              }
              if (stepKind === 'tool_result') {
                let updated = false;
                const toolCalls = [...prev.toolCalls]
                  .reverse()
                  .map((tc) => {
                    if (!updated && !tc.done && tc.tool_name === e.tool) {
                      updated = true;
                      return {
                        ...tc,
                        tool_output: e.output ?? e.output_preview ?? e.outputPreview ?? '',
                        endTimestamp: e.timestamp,
                        done: true,
                      };
                    }
                    return tc;
                  })
                  .reverse();
                return { ...prev, toolCalls };
              }
              return prev;
            });
          } else if (event.type === 'answer_chunk') {
            updateLastQaTurn((prev) => ({ ...prev, answer: (prev.answer ?? '') + event.chunk }));
          } else if (event.type === 'ask_complete') {
            updateLastQaTurn((prev) => ({
              ...prev,
              answer: event.answer ?? '',
              sources: (event.sources ?? []) as SourceReference[],
              loading: false,
              error: false,
            }));
          } else if (event.type === 'task_complete' && event.answer) {
            updateLastQaTurn((prev) => ({
              ...prev,
              answer: event.answer ?? '',
              sources: (event.sources ?? []) as SourceReference[],
              loading: false,
              error: false,
            }));
          } else if (event.type === 'ask_error') {
            updateLastQaTurn((prev) => ({ ...prev, answer: `Error: ${event.error}`, loading: false, error: true }));
          } else if (event.type === 'task_failed') {
            updateLastQaTurn((prev) => ({ ...prev, answer: `Error: ${event.error}`, loading: false, error: true }));
          }
        },
        () => {
          updateLastQaTurn((prev) => (prev.loading ? { ...prev, loading: false } : prev));
        },
        () => {
          updateLastQaTurn((prev) => ({
            ...prev,
            answer: 'Sorry, something went wrong. Please try again.',
            loading: false,
            error: true,
          }));
        },
      );
      cancelQaRef.current = cancel;
    }
  }, [qaInput, qaMode, projectId, updateLastQaTurn]);

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

      {/* Ask This Project panel */}
      {wikis.length > 0 && (
        <Box sx={{ mt: 5 }}>
          <Divider sx={{ mb: 3 }} />
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <QuestionAnswerOutlinedIcon color="action" fontSize="small" />
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
              Ask This Project
            </Typography>
          </Box>

          {/* Q&A conversation thread */}
          {qaTurns.length > 0 && (
            <Box sx={{ mb: 3 }}>
              {qaTurns.map((turn, i) => (
                <Box key={i} sx={{ mb: 3 }}>
                  {i > 0 && <Divider sx={{ mb: 3 }} />}
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    {/* Answer + sources */}
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <AnswerView
                        question={turn.question}
                        answer={turn.answer}
                        loading={turn.loading}
                        mode="light"
                      />
                      {/* Attribution chips + sources for completed turns */}
                      {!turn.loading && turn.sources.length > 0 && (
                        <Box sx={{ px: { xs: 3, md: 5 }, pb: 2 }}>
                          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                            Sources:
                          </Typography>
                          {turn.sources.map((src, j) => (
                            <Box
                              key={j}
                              sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.25, flexWrap: 'wrap' }}
                            >
                              {src.wiki_id && (
                                <Chip
                                  size="small"
                                  label={src.wiki_title ?? src.wiki_id}
                                  variant="outlined"
                                  color="primary"
                                  sx={{ fontSize: '0.65rem', height: 18 }}
                                />
                              )}
                              <Typography
                                variant="caption"
                                sx={{ fontFamily: 'monospace', color: 'text.secondary' }}
                              >
                                {src.file_path}
                                {src.line_start ? `:${src.line_start}` : ''}
                                {src.line_end ? `-${src.line_end}` : ''}
                              </Typography>
                            </Box>
                          ))}
                        </Box>
                      )}
                    </Box>
                    {/* Tool calls panel for the latest turn */}
                    {i === qaTurns.length - 1 && turn.toolCalls.length > 0 && (
                      <Box sx={{ width: 280, flexShrink: 0, display: { xs: 'none', md: 'block' } }}>
                        <ToolCallPanel toolCalls={turn.toolCalls} todos={turn.todos} />
                      </Box>
                    )}
                  </Box>
                </Box>
              ))}
              <Button
                size="small"
                variant="text"
                onClick={() => {
                  cancelQaRef.current?.();
                  cancelQaRef.current = null;
                  setQaTurns([]);
                }}
                sx={{ mb: 2, color: 'text.secondary' }}
              >
                Clear conversation
              </Button>
            </Box>
          )}

          {/* Input row */}
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-start' }}>
            <ToggleButtonGroup
              value={qaMode}
              exclusive
              onChange={(_, v) => { if (v) setQaMode(v); }}
              disabled={qaTurns[qaTurns.length - 1]?.loading}
              size="small"
              sx={{ flexShrink: 0 }}
            >
              <ToggleButton value="fast" sx={{ py: 0.5, px: 1.5, fontSize: 11, gap: 0.5 }}>
                <BoltIcon sx={{ fontSize: 14 }} /> Fast
              </ToggleButton>
              <ToggleButton value="deep" sx={{ py: 0.5, px: 1.5, fontSize: 11, gap: 0.5 }}>
                <SearchIcon sx={{ fontSize: 14 }} /> Deep
              </ToggleButton>
            </ToggleButtonGroup>
            <TextField
              value={qaInput}
              onChange={(e) => setQaInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleQaSubmit();
                }
              }}
              placeholder={
                qaTurns.length > 0
                  ? 'Ask a follow-up question…'
                  : qaMode === 'deep'
                  ? 'Ask a deep question across all project wikis…'
                  : 'Ask a question across all project wikis…'
              }
              size="small"
              fullWidth
              multiline
              maxRows={4}
              disabled={qaTurns[qaTurns.length - 1]?.loading}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={handleQaSubmit}
                      disabled={!qaInput.trim() || qaTurns[qaTurns.length - 1]?.loading}
                      size="small"
                      color="primary"
                    >
                      <SendIcon fontSize="small" />
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />
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
