import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Box, Typography, CircularProgress, Dialog, DialogTitle,
  DialogContent, DialogActions, Button, TextField,
} from '@mui/material';
import { useParams, useSearchParams } from 'react-router-dom';
import { WikiSidebar } from '../components/WikiSidebar';
import { WikiPageView } from '../components/WikiPageView';
import { OnThisPage } from '../components/OnThisPage';
import { useRepoContext } from '../context/RepoContext';
import { AskBar } from '../components/AskBar';
import { AnswerView } from '../components/AnswerView';
import { AnswerHeader } from '../components/AnswerHeader';
import { ToolCallPanel } from '../components/ToolCallPanel';
import { GenerationProgress } from '../components/GenerationProgress';
import { getWiki } from '../api/wiki';
import { subscribeSSE, subscribeResearchSSE, subscribeAskSSE } from '../api/sse';
import type { SSEEventData, ToolCallRecord, TodoItem } from '../api/sse';
import type { WikiPage } from '../components/WikiSidebar';
import type { WikiDetail } from '../api/wiki';
import type { components } from '../api/types.generated';

type SourceReference = components['schemas']['SourceReference'];

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
  mode: 'fast' | 'deep';
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
  const { setRepo, clearRepo } = useRepoContext();
  const [askState, setAskState] = useState<AskState | null>(null);
  const [searchParams, setSearchParams] = useSearchParams();
  const [genEvents, setGenEvents] = useState<SSEEventData[]>([]);
  const cancelResearchRef = useRef<(() => void) | null>(null);
  const [tokenDialogOpen, setTokenDialogOpen] = useState(false);
  const [tokenInput, setTokenInput] = useState('');

  // Track generating state from both URL params and API response
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
        } else if (data.status === 'failed' || data.status === 'partial' || data.status === 'cancelled') {
          // Show error state with repo info and retry button
          setIsGenerating(true);
          setActiveInvocationId(null);
          const errorMsg = (data as WikiDetail & { error?: string }).error ?? `Generation ${data.status}`;
          setGenEvents([{ type: 'task_failed' as const, taskId: data.invocation_id, status: 'failed', error: errorMsg }]);
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
        if (event.type === 'error' || event.type === 'task_failed' || event.type === 'task_cancelled') {
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
      setAskState(null);
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
    (question: string, useDeepResearch: boolean) => {
      if (!wikiId) return;

      // Cancel any in-flight research stream
      cancelResearchRef.current?.();
      cancelResearchRef.current = null;

      setAskState({
        question,
        answer: null,
        sources: [],
        toolCalls: [],
        todos: [],
        loading: true,
        mode: useDeepResearch ? 'deep' : 'fast',
      });

      if (useDeepResearch) {
        const cancel = subscribeResearchSSE(
          '/api/v1/research',
          { wiki_id: wikiId, question, research_type: 'general', enable_subagents: true },
          (event) => {
            if (event.type === 'thinking_step') {
              const e = event;
              // Accept both legacy step_type and new MCP stepType
              const stepKind = e.step_type ?? e.stepType;
              setAskState((prev) => {
                if (!prev) return prev;
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
                  // Update the last unfinished call for this tool
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
              setAskState((prev) =>
                prev ? { ...prev, answer: (prev.answer ?? '') + event.chunk } : null,
              );
            } else if (event.type === 'research_complete') {
              setAskState((prev) =>
                prev ? { ...prev, answer: event.report, loading: false } : null,
              );
            } else if (event.type === 'research_error') {
              setAskState((prev) =>
                prev ? { ...prev, answer: `Error: ${event.error}`, loading: false } : null,
              );
            } else if (event.type === 'task_failed') {
              // New MCP terminal failure event for research stream
              setAskState((prev) =>
                prev ? { ...prev, answer: `Error: ${event.error}`, loading: false } : null,
              );
            } else if (event.type === 'todo_update') {
              const todos = (event.todos as TodoItem[]) ?? [];
              setAskState((prev) => (prev ? { ...prev, todos } : null));
            }
          },
          () => {
            // Stream ended without research_complete — mark done
            setAskState((prev) => (prev?.loading ? { ...prev, loading: false } : prev));
          },
          () => {
            setAskState((prev) =>
              prev
                ? {
                    ...prev,
                    answer: 'Sorry, something went wrong. Please try again.',
                    loading: false,
                  }
                : null,
            );
          },
        );
        cancelResearchRef.current = cancel;
      } else {
        const cancel = subscribeAskSSE(
          '/api/v1/ask',
          { wiki_id: wikiId, question, chat_history: [], k: 15 },
          (event) => {
            if (event.type === 'thinking_step') {
              const e = event;
              // Accept both legacy step_type and new MCP stepType
              const stepKind = e.step_type ?? e.stepType;
              setAskState((prev) => {
                if (!prev) return prev;
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
              setAskState((prev) =>
                prev ? { ...prev, answer: (prev.answer ?? '') + event.chunk } : null,
              );
            } else if (event.type === 'ask_complete') {
              setAskState((prev) =>
                prev
                  ? {
                      ...prev,
                      answer: event.answer,
                      sources: event.sources as typeof prev.sources,
                      loading: false,
                    }
                  : null,
              );
            } else if (event.type === 'ask_error') {
              setAskState((prev) =>
                prev ? { ...prev, answer: `Error: ${event.error}`, loading: false } : null,
              );
            } else if (event.type === 'task_failed') {
              // New MCP terminal failure event for ask stream
              setAskState((prev) =>
                prev ? { ...prev, answer: `Error: ${event.error}`, loading: false } : null,
              );
            }
          },
          () => {
            setAskState((prev) => (prev?.loading ? { ...prev, loading: false } : prev));
          },
          () => {
            setAskState((prev) =>
              prev
                ? {
                    ...prev,
                    answer: 'Sorry, something went wrong. Please try again.',
                    loading: false,
                  }
                : null,
            );
          },
        );
        cancelResearchRef.current = cancel;
      }
    },
    [wikiId],
  );

  // Cancel research stream on unmount
  useEffect(() => {
    return () => {
      cancelResearchRef.current?.();
    };
  }, []);

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
        {!askState && !isGenerating && (
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
          ) : askState ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
              <AnswerHeader
                question={askState.question}
                mode={askState.mode}
                answer={askState.answer}
                onBack={() => {
                  cancelResearchRef.current?.();
                  cancelResearchRef.current = null;
                  setAskState(null);
                }}
              />
              <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
                {/* Left: Answer */}
                <Box
                  sx={{
                    flex: 1,
                    overflow: 'auto',
                    ...thinScrollbar,
                    borderRight: '1px solid',
                    borderColor: 'divider',
                  }}
                >
                  <AnswerView
                    question={askState.question}
                    answer={askState.answer}
                    loading={askState.loading}
                    mode={mode}
                  />
                </Box>
                {/* Right: Tool Calls (both fast and deep) */}
                <Box
                  sx={{
                    width: { xs: 0, md: '40%' },
                    maxWidth: 400,
                    flexShrink: 0,
                    overflow: 'hidden',
                    display: { xs: 'none', md: 'block' },
                  }}
                >
                  <ToolCallPanel toolCalls={askState.toolCalls} todos={askState.todos} />
                </Box>
              </Box>
            </Box>
          ) : pages.length === 0 ? (
            <Box display="flex" justifyContent="center" alignItems="center" minHeight="40vh">
              <Typography color="text.secondary">No wiki pages found.</Typography>
            </Box>
          ) : (
            <WikiPageView content={content} mode={mode} onNavigate={handleSelectPage} />
          )}
        </Box>

        {!askState && !isGenerating && <OnThisPage contentRef={contentRef} />}
      </Box>

      {!isGenerating && (
        <AskBar
          onSubmit={handleAsk}
          disabled={askState?.loading}
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
      )}

      {/* Token dialog for retrying wikis that originally required auth */}
      <Dialog open={tokenDialogOpen} onClose={() => setTokenDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Access Token Required</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            This repository was originally generated with an access token. Please provide one to retry.
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
          <Button variant="contained" onClick={() => doRetry(tokenInput || undefined)} disabled={!tokenInput}>
            Retry
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
