import { useCallback, useEffect, useMemo, useRef, useState, lazy, Suspense } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Chip,
  Divider,
  IconButton,
  Tooltip,
} from '@mui/material';
import ClearIcon from '@mui/icons-material/Clear';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import EditOutlinedIcon from '@mui/icons-material/EditOutlined';
import { useParams, useSearchParams } from 'react-router-dom';
import { WikiSidebar } from '../components/WikiSidebar';
import { WikiPageView } from '../components/WikiPageView';
import { OnThisPage } from '../components/OnThisPage';
import { useRepoContext } from '../context/RepoContext';
import { AskBar } from '../components/AskBar';
import type { AskMode } from '../components/AskBar';
import { AnswerView } from '../components/AnswerView';
import { AnswerHeader } from '../components/AnswerHeader';
import { ToolCallPanel } from '../components/ToolCallPanel';
import { GenerationProgress } from '../components/GenerationProgress';
import { getWiki, updateWikiDescription } from '../api/wiki';
import { subscribeSSE, subscribeResearchSSE, subscribeAskSSE } from '../api/sse';
import type { SSEEventData, ToolCallRecord, TodoItem } from '../api/sse';
import type { WikiPage } from '../components/WikiSidebar';
import type { WikiDetail } from '../api/wiki';
import type { components } from '../api/types.generated';

const CodeMapTree = lazy(() => import('../components/CodeMapTree'));

type SourceReference = components['schemas']['SourceReference'];
type ChatMessage = components['schemas']['ChatMessage'];
type CodeMapData = components['schemas']['CodeMapData'];

interface WikiViewerPageProps {
  mode?: 'light' | 'dark';
}

interface AskState {
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

interface PageData {
  id: string;
  title: string;
  order: number;
  content: string;
  section?: string;
}

const thinScrollbar = {
  scrollbarWidth: 'thin',
  scrollbarColor: 'rgba(255,255,255,0.15) transparent',
  '&::-webkit-scrollbar': { width: 6 },
  '&::-webkit-scrollbar-track': { background: 'transparent' },
  '&::-webkit-scrollbar-thumb': {
    background: 'rgba(255,255,255,0.15)',
    borderRadius: 3,
    '&:hover': { background: 'rgba(255,255,255,0.25)' },
  },
} as const;

export function WikiViewerPage({ mode = 'dark' }: WikiViewerPageProps) {
  const { wikiId } = useParams<{ wikiId: string }>();
  const [wiki, setWiki] = useState<WikiDetail | null>(null);
  const [pages, setPages] = useState<PageData[]>([]);
  const [activePageId, setActivePageId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const threadEndRef = useRef<HTMLDivElement>(null);
  const { setRepo, clearRepo } = useRepoContext();
  const [chatTurns, setChatTurns] = useState<AskState[]>([]);
  const [searchParams, setSearchParams] = useSearchParams();
  const [genEvents, setGenEvents] = useState<SSEEventData[]>([]);
  const cancelResearchRef = useRef<(() => void) | null>(null);
  const [tokenDialogOpen, setTokenDialogOpen] = useState(false);
  const [tokenInput, setTokenInput] = useState('');

  // Inline description edit state
  const [editingDescription, setEditingDescription] = useState(false);
  const [editDescription, setEditDescription] = useState('');
  const [savingDescription, setSavingDescription] = useState(false);

  // Derive LLM context from successfully completed turns (no errors)
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

  // Helper: update the last turn in chatTurns
  const updateLastTurn = useCallback((updater: (prev: AskState) => AskState) => {
    setChatTurns((prev) => {
      if (!prev.length) return prev;
      return [...prev.slice(0, -1), updater(prev[prev.length - 1])];
    });
  }, []);

  // Reset conversation when switching wikis
  useEffect(() => {
    setChatTurns([]);
  }, [wikiId]);

  // Auto-scroll to the bottom of the thread when new content arrives
  useEffect(() => {
    if (chatTurns.length > 0) {
      threadEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [chatTurns]);
  const urlGenerating = searchParams.get('generating') === 'true';
  const urlInvocationId = searchParams.get('invocation');
  const [activeInvocationId, setActiveInvocationId] = useState<string | null>(urlInvocationId);
  const [isGenerating, setIsGenerating] = useState(urlGenerating);

  // Sync URL params → state when they change (e.g. after refresh button)
  useEffect(() => {
    if (urlGenerating && urlInvocationId && urlInvocationId !== activeInvocationId) {
      setIsGenerating(true);
      setActiveInvocationId(urlInvocationId);
      setGenEvents([]);
    }
  }, [urlGenerating, urlInvocationId]); // intentionally excludes activeInvocationId to avoid re-trigger loops

  // Fetch wiki data from API
  useEffect(() => {
    if (!wikiId) return;
    setLoading(true);
    setError(null);

    getWiki(wikiId)
      .then((data) => {
        setWiki(data);
        setPages(data.pages);
        setRepo({
          wikiId: data.wiki_id,
          repoUrl: data.repo_url,
          branch: data.branch,
          indexedAt: data.indexed_at ?? data.created_at,
          commitHash: data.commit_hash,
        });
        if (data.pages.length > 0) {
          const urlPage = searchParams.get('page');
          const matched = urlPage ? data.pages.find((p) => p.id === urlPage) : null;
          setActivePageId(matched ? matched.id : data.pages[0].id);
        }

        // Auto-detect generating/failed state from API response
        if (data.status === 'generating' && data.invocation_id) {
          setIsGenerating(true);
          setActiveInvocationId(data.invocation_id);
        } else if (
          data.status === 'failed' ||
          data.status === 'partial' ||
          data.status === 'cancelled'
        ) {
          // Show error state with repo info and retry button
          setIsGenerating(true);
          setActiveInvocationId(null);
          const errorMsg =
            (data as WikiDetail & { error?: string }).error ?? `Generation ${data.status}`;
          setGenEvents([
            {
              type: 'task_failed' as const,
              taskId: data.invocation_id,
              status: 'failed',
              error: errorMsg,
            },
          ]);
        }
      })
      .catch(() => {
        // If 404 and we have URL params indicating generation, stay in generating mode
        if (urlGenerating && urlInvocationId) {
          setIsGenerating(true);
          setActiveInvocationId(urlInvocationId);
        } else {
          setError('Failed to load wiki');
        }
        setPages([]);
      })
      .finally(() => setLoading(false));
    return () => clearRepo();
  }, [wikiId]);

  // Subscribe to SSE when generating (#306 fix: reconnects on mount)
  useEffect(() => {
    const invId = activeInvocationId;
    if (!isGenerating || !invId) return;

    const source = subscribeSSE(
      `/api/v1/invocations/${invId}/stream`,
      (event) => {
        setGenEvents((prev) => [...prev, event]);
        if (event.type === 'wiki_complete' || event.type === 'task_complete') {
          source.close();
          setIsGenerating(false);
          setActiveInvocationId(null);
          // Remove URL params
          setSearchParams({}, { replace: true });
          // Reload wiki data after generation completes
          setTimeout(() => {
            setGenEvents([]);
            if (wikiId) {
              getWiki(wikiId)
                .then((data) => {
                  setWiki(data);
                  setPages(data.pages);
                  setRepo({
                    wikiId: data.wiki_id,
                    repoUrl: data.repo_url,
                    branch: data.branch,
                    indexedAt: data.indexed_at ?? data.created_at,
                    commitHash: data.commit_hash,
                  });
                  if (data.pages.length > 0) setActivePageId(data.pages[0].id);
                })
                .catch(() => {});
            }
          }, 2000);
        }
        // Terminal failure/cancellation — stop generating but stay on page so user can see error + retry
        if (
          event.type === 'error' ||
          event.type === 'task_failed' ||
          event.type === 'task_cancelled'
        ) {
          source.close();
          setActiveInvocationId(null);
          setSearchParams({}, { replace: true });
          // Keep isGenerating=true so GenerationProgress stays visible with retry button
        }
      },
      () => {
        setGenEvents((prev) => [
          ...prev,
          // Keep legacy shape so GenerationProgress recognises it as an error event
          { type: 'error', event: 'error', error: 'Connection lost', recoverable: true },
        ]);
      },
    );

    return () => source.close();
  }, [isGenerating, activeInvocationId, setSearchParams, wikiId]);

  const handleSelectPage = useCallback(
    (pageId: string) => {
      setActivePageId(pageId);
      setChatTurns([]);
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set('page', pageId);
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const handleAsk = useCallback(
    (question: string, mode: AskMode) => {
      if (!wikiId) return;

      // Cancel any in-flight research stream
      cancelResearchRef.current?.();
      cancelResearchRef.current = null;

      // Append a new turn to the thread
      setChatTurns((prev) => [
        ...prev,
        {
          question,
          answer: null,
          sources: [],
          toolCalls: [],
          todos: [],
          loading: true,
          error: false,
          mode,
        },
      ]);

      if (mode === 'deep' || mode === 'codemap') {
        const researchType = mode === 'codemap' ? 'codemap' : 'general';
        const cancel = subscribeResearchSSE(
          '/api/v1/research',
          { wiki_id: wikiId, question, research_type: researchType, chat_history: convHistory },
          (event) => {
            if (event.type === 'thinking_step') {
              const e = event;
              const stepKind = e.step_type ?? e.stepType;
              updateLastTurn((prev) => {
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
              updateLastTurn((prev) => ({ ...prev, answer: (prev.answer ?? '') + event.chunk }));
            } else if (event.type === 'code_map_ready') {
              // Code map arrives early (before refine_answer finishes)
              const codeMap = (event as { type: string; code_map?: unknown }).code_map as
                | AskState['codeMap']
                | undefined;
              if (codeMap) {
                updateLastTurn((prev) => ({ ...prev, codeMap }));
              }
            } else if (event.type === 'research_complete') {
              const answer = event.report ?? '';
              const codeMap = event.code_map as AskState['codeMap'] | undefined;
              updateLastTurn((prev) => ({
                ...prev,
                answer,
                loading: false,
                error: false,
                ...(codeMap ? { codeMap } : {}),
              }));
            } else if (event.type === 'research_error') {
              updateLastTurn((prev) => ({
                ...prev,
                answer: `Error: ${event.error}`,
                loading: false,
                error: true,
              }));
            } else if (event.type === 'task_failed') {
              updateLastTurn((prev) => ({
                ...prev,
                answer: `Error: ${event.error}`,
                loading: false,
                error: true,
              }));
            } else if (event.type === 'todo_update') {
              const todos = (event.todos as TodoItem[]) ?? [];
              updateLastTurn((prev) => ({ ...prev, todos }));
            }
          },
          () => {
            updateLastTurn((prev) => (prev.loading ? { ...prev, loading: false } : prev));
          },
          () => {
            updateLastTurn((prev) => ({
              ...prev,
              answer: 'Sorry, something went wrong. Please try again.',
              loading: false,
              error: true,
            }));
          },
        );
        cancelResearchRef.current = cancel;
      } else {
        const cancel = subscribeAskSSE(
          '/api/v1/ask',
          { wiki_id: wikiId, question, chat_history: convHistory, k: 15 },
          (event) => {
            if (event.type === 'thinking_step') {
              const e = event;
              const stepKind = e.step_type ?? e.stepType;
              updateLastTurn((prev) => {
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
              updateLastTurn((prev) => ({ ...prev, answer: (prev.answer ?? '') + event.chunk }));
            } else if (event.type === 'ask_complete') {
              const answer = event.answer ?? '';
              const sources = (event.sources ?? []) as AskState['sources'];
              updateLastTurn((prev) => ({
                ...prev,
                answer,
                sources,
                loading: false,
                error: false,
              }));
            } else if (event.type === 'task_complete' && event.answer) {
              const answer = event.answer;
              const sources = (event.sources ?? []) as AskState['sources'];
              updateLastTurn((prev) => ({
                ...prev,
                answer,
                sources,
                loading: false,
                error: false,
              }));
            } else if (event.type === 'ask_error') {
              updateLastTurn((prev) => ({
                ...prev,
                answer: `Error: ${event.error}`,
                loading: false,
                error: true,
              }));
            } else if (event.type === 'task_failed') {
              updateLastTurn((prev) => ({
                ...prev,
                answer: `Error: ${event.error}`,
                loading: false,
                error: true,
              }));
            }
          },
          () => {
            updateLastTurn((prev) => (prev.loading ? { ...prev, loading: false } : prev));
          },
          () => {
            updateLastTurn((prev) => ({
              ...prev,
              answer: 'Sorry, something went wrong. Please try again.',
              loading: false,
              error: true,
            }));
          },
        );
        cancelResearchRef.current = cancel;
      }
    },
    [wikiId, convHistory, updateLastTurn],
  );

  // Cancel research stream on unmount
  useEffect(() => {
    return () => {
      cancelResearchRef.current?.();
    };
  }, []);

  // Scroll to hash anchor after page content renders
  useEffect(() => {
    const hash = window.location.hash;
    if (!hash) return;
    const id = hash.slice(1);
    const timer = setTimeout(() => {
      const el = document.getElementById(id);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 150);
    return () => clearTimeout(timer);
  }, [activePageId]);

  const doRetry = useCallback(
    (accessToken?: string) => {
      if (!wikiId) return;
      setGenEvents([]);
      setTokenDialogOpen(false);
      setTokenInput('');
      import('../api/wiki').then(({ refreshWiki }) =>
        refreshWiki(wikiId, accessToken)
          .then((resp) => {
            setIsGenerating(true);
            setActiveInvocationId(resp.invocation_id);
          })
          .catch(() => {
            setIsGenerating(false);
            setError('Failed to retry generation');
          }),
      );
    },
    [wikiId],
  );

  const handleEditDescription = useCallback(() => {
    setEditDescription(wiki?.description ?? '');
    setEditingDescription(true);
  }, [wiki]);

  const handleCancelDescription = useCallback(() => {
    setEditingDescription(false);
  }, []);

  const handleSaveDescription = useCallback(async () => {
    if (!wikiId) return;
    setSavingDescription(true);
    try {
      const updated = await updateWikiDescription(wikiId, editDescription.trim() || null);
      setWiki((prev) => (prev ? { ...prev, description: updated.description } : prev));
      setEditingDescription(false);
    } catch (err) {
      console.error('Failed to save description', err);
    } finally {
      setSavingDescription(false);
    }
  }, [wikiId, editDescription]);

  const activePage = pages.find((p) => p.id === activePageId);
  const content = activePage?.content ?? '';
  const sidebarPages: WikiPage[] = pages.map((p) => ({
    id: p.id,
    title: p.title,
    order: p.order,
    section: p.section,
  }));

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error && !isGenerating) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 64px)' }}>
      <Box
        sx={{
          display: 'flex',
          flex: 1,
          overflow: 'hidden',
          maxWidth: 1400,
          mx: 'auto',
          width: '100%',
        }}
      >
        {!chatTurns.length && !isGenerating && (
          <WikiSidebar
            pages={sidebarPages}
            activePageId={activePageId}
            onSelectPage={handleSelectPage}
          />
        )}

        <Box
          ref={contentRef}
          sx={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}
        >
          {isGenerating ? (
            <Box sx={{ overflow: 'auto', flex: 1, ...thinScrollbar }}>
              <GenerationProgress
                events={genEvents}
                onRetry={() => {
                  if (!wikiId) return;
                  // If the wiki required a token, prompt for it first
                  if (wiki?.requires_token) {
                    setTokenDialogOpen(true);
                    return;
                  }
                  doRetry();
                }}
              />
            </Box>
          ) : chatTurns.length > 0 ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
              {/* Header shows first question + back/clear actions */}
              <AnswerHeader
                question={chatTurns[0].question}
                mode={chatTurns[chatTurns.length - 1].mode}
                answer={chatTurns[chatTurns.length - 1].answer}
                turnCount={chatTurns.length}
                onBack={() => {
                  cancelResearchRef.current?.();
                  cancelResearchRef.current = null;
                  setChatTurns([]);
                }}
              />
              <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
                {/* Left: Scrollable conversation thread */}
                <Box
                  sx={{
                    flex: 1,
                    overflow: 'auto',
                    ...thinScrollbar,
                    borderRight: '1px solid',
                    borderColor: 'divider',
                    pb: 12,
                  }}
                >
                  {chatTurns.map((turn, i) => (
                    <Box key={i}>
                      {i > 0 && <Divider sx={{ my: 2 }} />}
                      <AnswerView
                        question={turn.question}
                        answer={turn.answer}
                        loading={turn.loading}
                        mode={mode}
                      />
                    </Box>
                  ))}
                  <div ref={threadEndRef} />
                </Box>
                {/* Right: Tool Calls / Code Map for the latest turn */}
                <Box
                  sx={{
                    width: { xs: 0, md: '40%' },
                    maxWidth: 400,
                    flexShrink: 0,
                    overflowY: 'auto',
                    display: { xs: 'none', md: 'block' },
                    ...thinScrollbar,
                  }}
                >
                  {/* Code map first (primary content when available) */}
                  {chatTurns[chatTurns.length - 1].codeMap && (
                    <Suspense
                      fallback={
                        <Box sx={{ p: 2, display: 'flex', justifyContent: 'center' }}>
                          <CircularProgress size={24} />
                        </Box>
                      }
                    >
                      <CodeMapTree data={chatTurns[chatTurns.length - 1].codeMap!} />
                    </Suspense>
                  )}
                  {/* Tool calls below */}
                  <ToolCallPanel
                    toolCalls={chatTurns[chatTurns.length - 1].toolCalls}
                    todos={chatTurns[chatTurns.length - 1].todos}
                  />
                </Box>
              </Box>
            </Box>
          ) : pages.length === 0 ? (
            <Box display="flex" justifyContent="center" alignItems="center" minHeight="40vh">
              <Typography color="text.secondary">No wiki pages found.</Typography>
            </Box>
          ) : (
            <>
              {/* Inline description editor — shown above wiki page content */}
              <Box sx={{ px: 3, pt: 2, pb: 0 }}>
                {editingDescription ? (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, maxWidth: 600 }}>
                    <TextField
                      value={editDescription}
                      onChange={(e) => setEditDescription(e.target.value)}
                      label="Description"
                      size="small"
                      fullWidth
                      multiline
                      minRows={2}
                      disabled={savingDescription}
                      autoFocus
                    />
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Button
                        size="small"
                        variant="contained"
                        startIcon={
                          savingDescription ? <CircularProgress size={14} /> : <CheckIcon />
                        }
                        onClick={handleSaveDescription}
                        disabled={savingDescription}
                      >
                        Save
                      </Button>
                      <Button
                        size="small"
                        startIcon={<CloseIcon />}
                        onClick={handleCancelDescription}
                        disabled={savingDescription}
                      >
                        Cancel
                      </Button>
                    </Box>
                  </Box>
                ) : (
                  <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 0.5 }}>
                    {wiki?.description ? (
                      <Typography variant="body2" color="text.secondary" sx={{ flex: 1 }}>
                        {wiki.description}
                      </Typography>
                    ) : (
                      <Typography
                        variant="body2"
                        color="text.disabled"
                        sx={{ flex: 1, fontStyle: 'italic' }}
                      >
                        No description
                      </Typography>
                    )}
                    {wiki?.is_owner && (
                      <Tooltip title="Edit description">
                        <IconButton
                          size="small"
                          onClick={handleEditDescription}
                          sx={{ flexShrink: 0 }}
                        >
                          <EditOutlinedIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    )}
                  </Box>
                )}
              </Box>
              <WikiPageView
                content={content}
                mode={mode}
                onNavigate={handleSelectPage}
                pages={pages.map((p) => ({ id: p.id, title: p.title }))}
                wikiId={wiki?.wiki_id}
                wikiTitle={wiki?.title}
                isWikiComplete={wiki?.status === 'complete'}
              />
            </>
          )}
        </Box>

        {!chatTurns.length && !isGenerating && <OnThisPage contentRef={contentRef} />}
      </Box>

      {!isGenerating && (
        <>
          {chatTurns.length > 0 && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 0.5 }}>
              <Chip
                size="small"
                icon={<ClearIcon fontSize="small" />}
                label={`Context active (${chatTurns.filter((t) => !t.loading && !t.error).length} turn${chatTurns.filter((t) => !t.loading && !t.error).length !== 1 ? 's' : ''})`}
                onClick={() => setChatTurns([])}
                onDelete={() => setChatTurns([])}
                color="primary"
                variant="outlined"
                sx={{ fontSize: '0.72rem', cursor: 'pointer' }}
              />
            </Box>
          )}
          <AskBar
            onSubmit={handleAsk}
            disabled={chatTurns[chatTurns.length - 1]?.loading}
            placeholder={chatTurns.length > 0 ? 'Ask a follow-up question…' : undefined}
            repoLabel={
              wiki?.repo_url
                ? (() => {
                    try {
                      const p = new URL(wiki.repo_url).pathname.split('/').filter(Boolean);
                      return p.length >= 2 ? `${p[0]}/${p[1]}` : undefined;
                    } catch {
                      return undefined;
                    }
                  })()
                : undefined
            }
          />
        </>
      )}

      {/* Token dialog for retrying wikis that originally required auth */}
      <Dialog
        open={tokenDialogOpen}
        onClose={() => setTokenDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Access Token Required</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            This repository was originally generated with an access token. Please provide one to
            retry.
          </Typography>
          <TextField
            autoFocus
            fullWidth
            label="Access Token"
            type="password"
            value={tokenInput}
            onChange={(e) => setTokenInput(e.target.value)}
            placeholder="ghp_xxxx or glpat-xxxx"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTokenDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={() => doRetry(tokenInput || undefined)}
            disabled={!tokenInput}
          >
            Retry
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
