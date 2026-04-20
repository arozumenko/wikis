import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Autocomplete,
  Box,
  Button,
  Card,
  CardActionArea,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  Grid,
  IconButton,
  Snackbar,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import EditOutlinedIcon from '@mui/icons-material/EditOutlined';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import PublicOutlinedIcon from '@mui/icons-material/PublicOutlined';
import RemoveCircleOutlineIcon from '@mui/icons-material/RemoveCircleOutline';
import { useNavigate, useParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { ConfirmDialog } from '../components/ConfirmDialog';
import { AskBar } from '../components/AskBar';
import type { AskMode } from '../components/AskBar';
import { AnswerHeader } from '../components/AnswerHeader';
import { AnswerView } from '../components/AnswerView';
import { ToolCallPanel } from '../components/ToolCallPanel';
import {
  getProject,
  updateProject,
  deleteProject,
  addWikiToProject,
  removeWikiFromProject,
  listProjectWikis,
  type ProjectResponse,
} from '../api/project';
import { subscribeAskSSE, subscribeResearchSSE } from '../api/sse';
import type { ToolCallRecord, TodoItem } from '../api/sse';
import { listWikis } from '../api/wiki';
import type { components } from '../api/types.generated';

const CodeMapTree = lazy(() => import('../components/CodeMapTree'));

type WikiSummary = components['schemas']['WikiSummary'];
type SourceReference = components['schemas']['SourceReference'];
type ChatMessage = components['schemas']['ChatMessage'];
type CodeMapData = components['schemas']['CodeMapData'];

interface QATurn {
  question: string;
  answer: string | null;
  sources: SourceReference[];
  toolCalls: ToolCallRecord[];
  todos: TodoItem[];
  loading: boolean;
  error: boolean;
  mode: AskMode;
  codeMap?: CodeMapData | null;
}

const CARD_GRADIENTS = [
  'linear-gradient(135deg, #A855F7, #6366F1)',
  'linear-gradient(135deg, #FACC15, #22C55E)',
  'linear-gradient(135deg, #E0C3FC, #F5A0E6)',
  'linear-gradient(135deg, #22D3EE, #3B82F6)',
  'linear-gradient(135deg, #F43F5E, #7C3AED)',
  'linear-gradient(135deg, #D9F99D, #34D399)',
  'linear-gradient(135deg, #C084FC, #6366F1)',
  'linear-gradient(135deg, #FB923C, #F43F5E)',
  'linear-gradient(135deg, #2DD4BF, #10B981)',
  'linear-gradient(135deg, #F472B6, #A855F7)',
];

function cardGradient(id: string): string {
  let hash = 0;
  for (let i = 0; i < id.length; i++) {
    hash = (hash * 31 + id.charCodeAt(i)) | 0;
  }
  return CARD_GRADIENTS[Math.abs(hash) % CARD_GRADIENTS.length];
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
  const [project, setProject] = useState<ProjectResponse | null>(null);
  const [wikis, setWikis] = useState<WikiSummary[]>([]);
  const [allWikis, setAllWikis] = useState<WikiSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Description modal state
  const [descModalOpen, setDescModalOpen] = useState(false);
  const [descDraft, setDescDraft] = useState('');
  const [descOriginal, setDescOriginal] = useState('');
  const [savingDesc, setSavingDesc] = useState(false);
  const [discardConfirmOpen, setDiscardConfirmOpen] = useState(false);
  const [descExpanded, setDescExpanded] = useState(false);

  // Delete project state
  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // Add wiki modal state
  const [addWikiModalOpen, setAddWikiModalOpen] = useState(false);
  const [wikiToAdd, setWikiToAdd] = useState<WikiSummary | null>(null);
  const [addingWiki, setAddingWiki] = useState(false);
  const [removingWikiId, setRemovingWikiId] = useState<string | null>(null);

  const [snackMessage, setSnackMessage] = useState<string | null>(null);
  const [snackSeverity, setSnackSeverity] = useState<'success' | 'error'>('error');

  // Q&A state — mirrors WikiViewerPage pattern exactly
  const [chatTurns, setChatTurns] = useState<QATurn[]>([]);
  const cancelStreamRef = useRef<(() => void) | null>(null);

  const updateLastTurn = useCallback((updater: (prev: QATurn) => QATurn) => {
    setChatTurns((prev) => {
      if (!prev.length) return prev;
      return [...prev.slice(0, -1), updater(prev[prev.length - 1])];
    });
  }, []);

  // Derive chat history from completed turns (for follow-up context)
  const convHistory = useMemo<ChatMessage[]>(
    () =>
      chatTurns
        .filter((t) => !t.loading && !t.error && t.answer !== null)
        .flatMap((t) => [
          { role: 'user' as const, content: t.question },
          { role: 'assistant' as const, content: t.answer! },
        ]),
    [chatTurns],
  );

  // Cancel any in-flight SSE stream on unmount
  useEffect(() => () => { cancelStreamRef.current?.(); }, []);

  // Warn on browser navigation (tab close / refresh) when description modal has unsaved changes.
  useEffect(() => {
    if (!descModalOpen) return;
    const handler = (e: BeforeUnloadEvent) => {
      if (descDraft !== descOriginal) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [descModalOpen, descDraft, descOriginal]);

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
      // Only fetch all wikis if current user is owner (needed for "add wiki" panel)
      if (proj.is_owner) {
        const allWikiRes = await listWikis();
        setAllWikis(allWikiRes.wikis ?? []);
      }
    } catch {
      setError('Failed to load project.');
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    load();
  }, [load]);

  const isOwner = project?.is_owner ?? false;

  // Wikis not yet in this project
  const memberIds = new Set(wikis.map((w) => w.wiki_id));
  const availableToAdd = allWikis.filter((w) => !memberIds.has(w.wiki_id));

  // Description modal handlers
  const handleOpenDescModal = useCallback(() => {
    const current = project?.description ?? '';
    setDescOriginal(current);
    setDescDraft(current);
    setDescModalOpen(true);
  }, [project]);

  const descHasChanges = descDraft !== descOriginal;

  const handleCloseDescModal = useCallback(() => {
    if (descHasChanges) {
      setDiscardConfirmOpen(true);
    } else {
      setDescModalOpen(false);
    }
  }, [descHasChanges]);

  const handleDiscardDesc = useCallback(() => {
    setDiscardConfirmOpen(false);
    setDescModalOpen(false);
  }, []);

  const handleSaveDesc = useCallback(async () => {
    if (!project) return;
    setSavingDesc(true);
    try {
      const trimmed = descDraft.trim();
      const updated = await updateProject(project.id, {
        description: trimmed || '',
      });
      setProject(updated);
      setDescModalOpen(false);
      showSnack('Description updated', 'success');
    } catch {
      showSnack('Failed to save description');
    } finally {
      setSavingDesc(false);
    }
  }, [project, descDraft, showSnack]);

  // CSS-based truncation: use a callback ref + ResizeObserver to detect overflow
  const [descOverflows, setDescOverflows] = useState(false);
  const DESC_COLLAPSED_HEIGHT = 280; // ~10 lines at normal line-height
  const descObserverRef = useRef<ResizeObserver | null>(null);
  const descContentRef = useCallback((el: HTMLDivElement | null) => {
    // Disconnect previous observer
    descObserverRef.current?.disconnect();
    descObserverRef.current = null;
    if (!el) return;
    const check = () => setDescOverflows(el.scrollHeight > DESC_COLLAPSED_HEIGHT + 10);
    requestAnimationFrame(check);
    const observer = new ResizeObserver(check);
    observer.observe(el);
    descObserverRef.current = observer;
  }, []);
  // Cleanup observer on unmount
  useEffect(() => () => { descObserverRef.current?.disconnect(); }, []);

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

  // Shared SSE event handler for thinking_step events (identical to WikiViewerPage)
  const handleThinkingStep = useCallback((event: Record<string, unknown>) => {
    const stepKind = (event.step_type ?? event.stepType) as string | undefined;
    updateLastTurn((prev) => {
      if (stepKind === 'tool_call') {
        const record: ToolCallRecord = {
          tool_name: event.tool as string,
          tool_input: (event.input as string) ?? '',
          tool_output: null,
          timestamp: event.timestamp as string,
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
            if (!updated && !tc.done && tc.tool_name === (event.tool as string)) {
              updated = true;
              return {
                ...tc,
                tool_output: ((event.output ?? event.output_preview ?? event.outputPreview) as string) ?? '',
                endTimestamp: event.timestamp as string,
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
  }, [updateLastTurn]);

  const handleAsk = useCallback(
    (question: string, mode: AskMode) => {
      if (!projectId) return;

      cancelStreamRef.current?.();
      cancelStreamRef.current = null;

      setChatTurns((prev) => [
        ...prev,
        { question, answer: null, sources: [], toolCalls: [], todos: [], loading: true, error: false, mode },
      ]);

      if (mode === 'deep' || mode === 'codemap') {
        const researchType = mode === 'codemap' ? 'codemap' : 'general';
        const cancel = subscribeResearchSSE(
          '/api/v1/research',
          { project_id: projectId, question, research_type: researchType, chat_history: convHistory },
          (event) => {
            if (event.type === 'thinking_step') {
              handleThinkingStep(event as unknown as Record<string, unknown>);
            } else if (event.type === 'answer_chunk') {
              updateLastTurn((prev) => ({ ...prev, answer: (prev.answer ?? '') + event.chunk }));
            } else if (event.type === 'code_map_ready') {
              const codeMap = (event as { type: string; code_map?: unknown }).code_map as CodeMapData | undefined;
              if (codeMap) {
                updateLastTurn((prev) => ({ ...prev, codeMap }));
              }
            } else if (event.type === 'research_complete') {
              const codeMap = event.code_map as CodeMapData | undefined;
              updateLastTurn((prev) => ({
                ...prev,
                answer: event.report ?? '',
                loading: false,
                error: false,
                ...(codeMap ? { codeMap } : {}),
              }));
            } else if (event.type === 'research_error') {
              updateLastTurn((prev) => ({ ...prev, answer: `Error: ${event.error}`, loading: false, error: true }));
            } else if (event.type === 'task_failed') {
              updateLastTurn((prev) => ({ ...prev, answer: `Error: ${event.error}`, loading: false, error: true }));
            } else if (event.type === 'todo_update') {
              const todos = (event.todos as TodoItem[]) ?? [];
              updateLastTurn((prev) => ({ ...prev, todos }));
            }
          },
          () => { updateLastTurn((prev) => (prev.loading ? { ...prev, loading: false } : prev)); },
          () => { updateLastTurn((prev) => ({ ...prev, answer: 'Sorry, something went wrong. Please try again.', loading: false, error: true })); },
        );
        cancelStreamRef.current = cancel;
      } else {
        const cancel = subscribeAskSSE(
          '/api/v1/ask',
          { project_id: projectId, question, chat_history: convHistory, k: 15 },
          (event) => {
            if (event.type === 'thinking_step') {
              handleThinkingStep(event as unknown as Record<string, unknown>);
            } else if (event.type === 'answer_chunk') {
              updateLastTurn((prev) => ({ ...prev, answer: (prev.answer ?? '') + event.chunk }));
            } else if (event.type === 'ask_complete') {
              updateLastTurn((prev) => ({ ...prev, answer: event.answer ?? '', sources: (event.sources ?? []) as SourceReference[], loading: false, error: false }));
            } else if (event.type === 'task_complete' && event.answer) {
              updateLastTurn((prev) => ({ ...prev, answer: event.answer ?? '', sources: (event.sources ?? []) as SourceReference[], loading: false, error: false }));
            } else if (event.type === 'ask_error') {
              updateLastTurn((prev) => ({ ...prev, answer: `Error: ${event.error}`, loading: false, error: true }));
            } else if (event.type === 'task_failed') {
              updateLastTurn((prev) => ({ ...prev, answer: `Error: ${event.error}`, loading: false, error: true }));
            }
          },
          () => { updateLastTurn((prev) => (prev.loading ? { ...prev, loading: false } : prev)); },
          () => { updateLastTurn((prev) => ({ ...prev, answer: 'Sorry, something went wrong. Please try again.', loading: false, error: true })); },
        );
        cancelStreamRef.current = cancel;
      }
    },
    [projectId, convHistory, updateLastTurn, handleThinkingStep],
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

  const thinScrollbar = {
    scrollbarWidth: 'thin' as const,
    scrollbarColor: 'rgba(255,255,255,0.15) transparent',
    '&::-webkit-scrollbar': { width: 6 },
    '&::-webkit-scrollbar-track': { background: 'transparent' },
    '&::-webkit-scrollbar-thumb': { background: 'rgba(255,255,255,0.15)', borderRadius: 3, '&:hover': { background: 'rgba(255,255,255,0.25)' } },
  };

  // When conversation is active, show full-screen answer view (identical to WikiViewerPage)
  if (chatTurns.length > 0 && project) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
        <AnswerHeader
          question={chatTurns[0].question}
          mode={chatTurns[chatTurns.length - 1].mode}
          answer={chatTurns[chatTurns.length - 1].answer}
          turnCount={chatTurns.length}
          onBack={() => {
            cancelStreamRef.current?.();
            cancelStreamRef.current = null;
            setChatTurns([]);
          }}
        />
        {/* Scrollable conversation — each turn has its own answer + tool calls side by side */}
        <Box sx={{ flex: 1, overflow: 'auto', ...thinScrollbar, pb: 12 }}>
          {chatTurns.map((turn, i) => (
            <Box key={i}>
              {i > 0 && <Divider />}
              <Box sx={{ display: 'flex' }}>
                {/* Answer */}
                <Box sx={{ flex: 1, minWidth: 0, borderRight: '1px solid', borderColor: 'divider' }}>
                  <AnswerView
                    question={turn.question}
                    answer={turn.answer}
                    loading={turn.loading}
                    mode="dark"
                  />
                </Box>
                {/* Code Map + Tool Calls — per turn */}
                <Box sx={{ width: { xs: 0, md: '40%' }, maxWidth: 400, flexShrink: 0, display: { xs: 'none', md: 'block' }, overflowY: 'auto', ...thinScrollbar }}>
                  {turn.codeMap && (
                    <Suspense
                      fallback={
                        <Box sx={{ p: 2, display: 'flex', justifyContent: 'center' }}>
                          <CircularProgress size={24} />
                        </Box>
                      }
                    >
                      <CodeMapTree data={turn.codeMap} />
                    </Suspense>
                  )}
                  <ToolCallPanel
                    toolCalls={turn.toolCalls}
                    todos={turn.todos}
                  />
                </Box>
              </Box>
            </Box>
          ))}
        </Box>
        {/* AskBar — same component as wiki viewer */}
        <AskBar
          onSubmit={handleAsk}
          disabled={chatTurns[chatTurns.length - 1]?.loading}
          placeholder="Ask a follow-up question…"
        />
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
          <Alert onClose={() => setSnackMessage(null)} severity={snackSeverity} sx={{ width: '100%' }}>
            {snackMessage}
          </Alert>
        </Snackbar>
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
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
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
          {isOwner && (
            <Box sx={{ display: 'flex', gap: 0.5, flexShrink: 0 }}>
              <Tooltip title="Edit description">
                <IconButton size="small" onClick={handleOpenDescModal}>
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
      </Box>

      {/* Description section */}
      <Box sx={{ mb: 4 }}>
        {project.description ? (
          <Box>
            <Box
              ref={descContentRef}
              sx={(theme) => ({
                overflow: 'hidden',
                position: 'relative',
                maxHeight: descExpanded ? 'none' : DESC_COLLAPSED_HEIGHT,
                ...(!descExpanded && descOverflows ? {
                  '&::after': {
                    content: '""',
                    position: 'absolute',
                    bottom: 0,
                    left: 0,
                    right: 0,
                    height: 80,
                    background: `linear-gradient(transparent, ${theme.palette.background.default})`,
                    pointerEvents: 'none',
                  },
                } : {}),
                '& h1': {
                  fontFamily: '"Playfair Display", Georgia, serif',
                  mt: 0, mb: 2, fontSize: '1.75rem', letterSpacing: '-0.02em',
                },
                '& h2': {
                  fontFamily: '"Playfair Display", Georgia, serif',
                  mt: 4, mb: 1.5, pt: 2, fontSize: '1.35rem', letterSpacing: '-0.01em',
                  borderTop: '1px solid', borderColor: 'divider',
                },
                '& h3': {
                  fontFamily: '"Playfair Display", Georgia, serif',
                  mt: 3, mb: 1, fontSize: '1.15rem',
                },
                '& p': { mb: 2, lineHeight: 1.8 },
                '& ul, & ol': { mb: 2, pl: 3 },
                '& li': { mb: 0.5, lineHeight: 1.7 },
                '& table': {
                  width: '100%', borderCollapse: 'collapse', mb: 2, borderRadius: '8px', overflow: 'hidden',
                  '& th, & td': { border: '1px solid', borderColor: 'divider', px: 2, py: 1, textAlign: 'left' },
                  '& th': { fontWeight: 600, bgcolor: 'action.hover' },
                },
                '& pre': {
                  borderRadius: '12px', overflow: 'auto',
                  bgcolor: 'rgba(255,255,255,0.03)', p: 2, my: 2,
                  border: '1px solid', borderColor: 'divider',
                },
                '& code': { fontFamily: '"JetBrains Mono", "Fira Code", monospace', fontSize: '0.85em' },
                '& :not(pre) > code': {
                  bgcolor: 'rgba(255, 107, 74, 0.1)', color: 'primary.main',
                  px: 0.75, py: 0.25, borderRadius: '4px', fontWeight: 500,
                },
                '& a': { color: 'primary.main', textDecoration: 'none', '&:hover': { textDecoration: 'underline' } },
                '& img': { maxWidth: '100%', borderRadius: '8px' },
                '& hr': { border: 'none', borderTop: '1px solid', borderColor: 'divider', my: 3 },
              })}
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
                {project.description}
              </ReactMarkdown>
            </Box>
            {descOverflows && (
              <Button
                size="small"
                onClick={() => setDescExpanded((v) => !v)}
                sx={{ mt: 0.5, color: 'text.secondary', textTransform: 'none', fontSize: '0.8rem' }}
              >
                {descExpanded ? 'Show less' : 'Show more'}
              </Button>
            )}
          </Box>
        ) : (
          <Typography
            variant="body2"
            sx={{ color: 'text.disabled', fontStyle: 'italic', cursor: 'pointer' }}
            onClick={handleOpenDescModal}
          >
            No description
          </Typography>
        )}
      </Box>

      {/* Wiki grid */}
      {wikis.length === 0 && !isOwner ? (
        <Alert severity="info">
          No wikis in this project yet.
        </Alert>
      ) : (
        <Grid container spacing={2.5} sx={{ mb: 4 }}>
          {isOwner && (
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
                  onClick={() => setAddWikiModalOpen(true)}
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
          {wikis.map((wiki) => {
            const gradient = cardGradient(wiki.wiki_id);
            return (
              <Grid item xs={12} sm={6} md={4} key={wiki.wiki_id}>
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
                      borderColor: 'divider',
                      transition: 'transform 0.2s ease',
                    }}
                  >
                    <Box
                      onClick={() => navigate(`/wiki/${wiki.wiki_id}`)}
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
                        <Typography variant="body1" sx={{ fontWeight: 600, mb: 0.5 }} noWrap>
                          {extractOwnerRepo(wiki.repo_url)}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {wiki.branch}
                          {wiki.page_count > 0 ? ` · ${wiki.page_count} pages` : ''}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mt: 0.5 }}>
                          <Chip
                            icon={
                              (wiki.visibility ?? 'personal') === 'personal' ? (
                                <LockOutlinedIcon sx={{ fontSize: '0.75rem !important' }} />
                              ) : (
                                <PublicOutlinedIcon sx={{ fontSize: '0.75rem !important' }} />
                              )
                            }
                            label={(wiki.visibility ?? 'personal') === 'personal' ? 'Personal' : 'Shared'}
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
                          <Tooltip title="Remove from project">
                            <span>
                              <IconButton
                                size="small"
                                disabled={removingWikiId === wiki.wiki_id}
                                onClick={() => handleRemoveWiki(wiki.wiki_id)}
                                sx={{
                                  color: 'text.secondary',
                                  '&:hover': { color: 'error.main' },
                                }}
                              >
                                {removingWikiId === wiki.wiki_id ? (
                                  <CircularProgress size={14} />
                                ) : (
                                  <RemoveCircleOutlineIcon fontSize="small" />
                                )}
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
        </Grid>
      )}

      {/* AskBar — same component as wiki viewer */}
      {wikis.length > 0 && (
        <AskBar
          onSubmit={handleAsk}
          placeholder="Ask a question across all project wikis…"
        />
      )}

      {/* Edit Description Modal */}
      <Dialog
        open={descModalOpen}
        onClose={handleCloseDescModal}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: { bgcolor: 'background.default', backgroundImage: 'none', minHeight: '50vh' },
        }}
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          Edit Description
          <IconButton size="small" onClick={handleCloseDescModal}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </DialogTitle>
        <DialogContent sx={{ pt: '8px !important' }}>
          <TextField
            value={descDraft}
            onChange={(e) => setDescDraft(e.target.value)}
            placeholder="Write your project description in Markdown..."
            multiline
            fullWidth
            minRows={12}
            maxRows={20}
            disabled={savingDesc}
            InputProps={{
              sx: {
                fontFamily: '"JetBrains Mono", "Fira Code", monospace',
                fontSize: '0.9rem',
                lineHeight: 1.7,
                '& textarea': { p: 0 },
              },
            }}
            sx={{
              '& .MuiOutlinedInput-root': {
                bgcolor: 'rgba(255,255,255,0.02)',
                p: 2.5,
                borderRadius: 2,
              },
            }}
          />
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={handleCloseDescModal} disabled={savingDesc}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleSaveDesc}
            disabled={savingDesc || !descHasChanges}
            startIcon={savingDesc ? <CircularProgress size={14} /> : <CheckIcon />}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>

      {/* Add Repository Modal */}
      <Dialog
        open={addWikiModalOpen}
        onClose={() => setAddWikiModalOpen(false)}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: { bgcolor: 'background.default', backgroundImage: 'none' },
        }}
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          Add Repository
          <IconButton size="small" onClick={() => setAddWikiModalOpen(false)}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </DialogTitle>
        <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: '8px !important' }}>
          {wikis.length > 0 && (
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Current wikis
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {wikis.map((wiki) => {
                  const isRemoving = removingWikiId === wiki.wiki_id;
                  return (
                    <Box
                      key={wiki.wiki_id}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1,
                        px: 1.5,
                        py: 0.75,
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 2,
                        minWidth: 0,
                        flex: { xs: '0 1 100%', sm: '0 1 calc(50% - 8px)' },
                      }}
                    >
                      <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Typography variant="caption" sx={{ fontWeight: 600, display: 'block' }} noWrap>
                          {extractOwnerRepo(wiki.repo_url)}
                        </Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                          {wiki.branch}
                        </Typography>
                      </Box>
                      <Tooltip title="Remove from project">
                        <span>
                          <IconButton
                            size="small"
                            disabled={isRemoving}
                            onClick={() => handleRemoveWiki(wiki.wiki_id)}
                            sx={{ color: 'text.disabled', '&:hover': { color: 'error.main' }, p: 0.25 }}
                          >
                            {isRemoving ? (
                              <CircularProgress size={12} />
                            ) : (
                              <RemoveCircleOutlineIcon sx={{ fontSize: 16 }} />
                            )}
                          </IconButton>
                        </span>
                      </Tooltip>
                    </Box>
                  );
                })}
              </Box>
            </Box>
          )}
          {availableToAdd.length > 0 ? (
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Add a wiki
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
                    <TextField {...params} placeholder="Search wikis to add..." />
                  )}
                  isOptionEqualToValue={(a, b) => a.wiki_id === b.wiki_id}
                />
                <Button
                  variant="outlined"
                  size="small"
                  disabled={!wikiToAdd || addingWiki}
                  onClick={handleAddWiki}
                  startIcon={addingWiki ? <CircularProgress size={14} /> : undefined}
                >
                  Add
                </Button>
              </Box>
            </Box>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No more wikis available to add.
            </Typography>
          )}
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setAddWikiModalOpen(false)}>
            Done
          </Button>
        </DialogActions>
      </Dialog>

      {/* Discard changes confirmation */}
      <ConfirmDialog
        open={discardConfirmOpen}
        title="Unsaved Changes"
        message="You have unsaved changes to the description. Discard them?"
        confirmLabel="Discard"
        onConfirm={handleDiscardDesc}
        onCancel={() => setDiscardConfirmOpen(false)}
      />

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
