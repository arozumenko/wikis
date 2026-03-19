import { useEffect, useMemo, useRef } from 'react';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Step,
  StepContent,
  StepLabel,
  Stepper,
  Typography,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import DescriptionOutlinedIcon from '@mui/icons-material/DescriptionOutlined';
import SearchIcon from '@mui/icons-material/Search';
import EditNoteIcon from '@mui/icons-material/EditNote';
import type { SSEEventData } from '../api/sse';

/* ------------------------------------------------------------------ */
/*  Stepper step definitions                                          */
/* ------------------------------------------------------------------ */

interface StepDef {
  key: string;
  label: string;
  /** Backend _phase values that map to this step */
  phases: string[];
}

const STEPS: StepDef[] = [
  { key: 'clone', label: 'Clone & index repository', phases: ['configuring', 'indexing'] },
  { key: 'plan', label: 'Plan wiki structure', phases: ['planning'] },
  { key: 'generate', label: 'Generate content', phases: ['generating'] },
  { key: 'finalize', label: 'Review & finalize', phases: ['enhancing', 'quality', 'storing'] },
];

/** Map a backend phase string to a stepper step index. */
function phaseToStep(phase: string): number {
  for (let i = 0; i < STEPS.length; i++) {
    if (STEPS[i].phases.includes(phase)) return i;
  }
  return -1;
}

/* ------------------------------------------------------------------ */
/*  Event helpers (backward-compatible)                               */
/* ------------------------------------------------------------------ */

interface GenerationProgressProps {
  events: SSEEventData[];
  onRetry?: () => void;
}

/** Resolve phase from both old (`phase`) and new (`_phase`) formats. */
function resolvePhase(event: Extract<SSEEventData, { type: 'progress' }>): string {
  return (event._phase ?? (event as unknown as { phase?: string }).phase) ?? '';
}

/* ------------------------------------------------------------------ */
/*  Log entries for the right panel                                   */
/* ------------------------------------------------------------------ */

interface LogEntry {
  icon: 'check' | 'spinner' | 'error' | 'info' | 'page' | 'search' | 'write';
  text: string;
}

/** Classify a progress message to pick a meaningful icon. */
function classifyMessage(msg: string): LogEntry['icon'] {
  if (/Generated page:/i.test(msg)) return 'page';
  if (/Retriev|Discover|Search|context for/i.test(msg)) return 'search';
  if (/Writing:|Generating \d|Enhancing:|Assessing quality/i.test(msg)) return 'write';
  return 'spinner';
}

function buildLog(events: SSEEventData[]): LogEntry[] {
  const log: LogEntry[] = [];
  let lastMessage = '';

  for (const event of events) {
    if (event.type === 'progress') {
      const msg = event.message ?? '';
      // Show every distinct progress message — this is the verbose activity feed
      if (msg && msg !== lastMessage) {
        // "Generated page: X (N/M)" → completed page
        const pageMatch = msg.match(/Generated page:\s*(.+)\s*\(\d+\/\d+\)/);
        if (pageMatch) {
          log.push({ icon: 'page', text: pageMatch[1].trim() });
        } else {
          // Mark previous spinner as done when a new status arrives
          for (let j = log.length - 1; j >= 0; j--) {
            if (log[j].icon === 'spinner') {
              log[j].icon = 'check';
              break;
            }
          }
          log.push({ icon: classifyMessage(msg), text: msg });
        }
        lastMessage = msg;
      }
    } else if (event.type === 'page_complete') {
      const title =
        event._pageTitle ??
        (event as unknown as { page_title?: string }).page_title ??
        'Unknown page';
      log.push({ icon: 'check', text: title });
    } else if (event.type === 'wiki_complete') {
      for (const entry of log) {
        if (entry.icon === 'spinner') entry.icon = 'check';
      }
      const e = event as Extract<SSEEventData, { type: 'wiki_complete' }>;
      log.push({ icon: 'check', text: `Done — ${e.page_count} pages in ${e.execution_time.toFixed(1)}s` });
    } else if (event.type === 'task_complete') {
      for (const entry of log) {
        if (entry.icon === 'spinner') entry.icon = 'check';
      }
      const e = event as Extract<SSEEventData, { type: 'task_complete' }>;
      const pageCount = e.pageCount ?? e.page_count ?? 0;
      const execTime = e.executionTime ?? e.execution_time;
      const timeStr = execTime != null ? ` in ${execTime.toFixed(1)}s` : '';
      log.push({ icon: 'check', text: `Done — ${pageCount} pages${timeStr}` });
    } else if (event.type === 'error') {
      log.push({ icon: 'error', text: (event as Extract<SSEEventData, { type: 'error' }>).error });
    } else if (event.type === 'task_failed') {
      log.push({ icon: 'error', text: (event as Extract<SSEEventData, { type: 'task_failed' }>).error });
    } else if (event.type === 'task_cancelled') {
      log.push({ icon: 'info', text: (event as Extract<SSEEventData, { type: 'task_cancelled' }>).statusMessage ?? 'Generation cancelled' });
    } else if (event.type === 'retry') {
      const e = event as Extract<SSEEventData, { type: 'retry' }>;
      log.push({ icon: 'info', text: `Retrying (attempt ${e.attempt}/${e.max_attempts})...` });
    } else if (event.type === 'fallback') {
      const e = event as Extract<SSEEventData, { type: 'fallback' }>;
      log.push({ icon: 'info', text: `Switched to ${e.to_model}` });
    } else if (event.type === 'message') {
      const e = event as Extract<SSEEventData, { type: 'message' }>;
      if (e.attempt != null) {
        const maxAttempts = e.maxAttempts ?? e.max_attempts;
        log.push({
          icon: 'info',
          text: maxAttempts != null
            ? `Retrying (attempt ${e.attempt}/${maxAttempts})...`
            : `Retrying (attempt ${e.attempt})...`,
        });
      } else if (e.toModel ?? e.to_model) {
        log.push({ icon: 'info', text: `Switched to ${e.toModel ?? e.to_model}` });
      }
    }
  }

  return log;
}

/* ------------------------------------------------------------------ */
/*  Derived state from SSE events                                     */
/* ------------------------------------------------------------------ */

interface DerivedState {
  activeStep: number;
  stepMessage: string;
  pagesCompleted: number;
  pagesTotal: number | null;
  isComplete: boolean;
  completionSummary: string | null;
  errorEvent: Extract<SSEEventData, { type: 'error' | 'task_failed' | 'task_cancelled' }> | null;
  errorStep: number;
}

function deriveState(events: SSEEventData[]): DerivedState {
  let activeStep = -1;
  let stepMessage = '';
  let pagesCompleted = 0;
  let pagesTotal: number | null = null;
  let isComplete = false;
  let completionSummary: string | null = null;
  let errorEvent: DerivedState['errorEvent'] = null;
  let errorStep = -1;

  for (const event of events) {
    if (event.type === 'progress') {
      const phase = resolvePhase(event);
      const step = phaseToStep(phase);
      if (step >= 0) {
        activeStep = step;
        stepMessage = event.message ?? '';
      }
      if (phase === 'generating' && event.message) {
        const initMatch = event.message.match(/Generating (\d+) wiki pages/);
        if (initMatch) {
          pagesTotal = parseInt(initMatch[1], 10);
          pagesCompleted = 0;
        }
        const pageMatch = event.message.match(/\((\d+)\/(\d+)\)/);
        if (pageMatch) {
          pagesCompleted = parseInt(pageMatch[1], 10);
          pagesTotal = parseInt(pageMatch[2], 10);
        }
      }
    } else if (event.type === 'page_complete') {
      // pagesCompleted already tracked via progress message regex during generating phase
      // page_complete events fire during storing phase — don't double-count
    } else if (event.type === 'wiki_complete') {
      isComplete = true;
      activeStep = STEPS.length;
      const e = event as Extract<SSEEventData, { type: 'wiki_complete' }>;
      completionSummary = `${e.page_count} pages generated in ${e.execution_time.toFixed(1)}s`;
    } else if (event.type === 'task_complete') {
      isComplete = true;
      activeStep = STEPS.length;
      const e = event as Extract<SSEEventData, { type: 'task_complete' }>;
      const pageCount = e.pageCount ?? e.page_count ?? 0;
      const execTime = e.executionTime ?? e.execution_time;
      completionSummary =
        execTime != null
          ? `${pageCount} pages generated in ${execTime.toFixed(1)}s`
          : `${pageCount} pages generated`;
    } else if (event.type === 'error' || event.type === 'task_failed' || event.type === 'task_cancelled') {
      errorEvent = event as DerivedState['errorEvent'];
      errorStep = Math.max(activeStep, 0);
    }
  }

  return { activeStep, stepMessage, pagesCompleted, pagesTotal, isComplete, completionSummary, errorEvent, errorStep };
}

/* ------------------------------------------------------------------ */
/*  Step icon                                                         */
/* ------------------------------------------------------------------ */

function StepIcon({ stepIndex, activeStep, errorStep, isComplete }: {
  stepIndex: number;
  activeStep: number;
  errorStep: number;
  isComplete: boolean;
}) {
  if (errorStep === stepIndex && !isComplete) {
    return <ErrorOutlineIcon sx={{ color: 'error.main' }} />;
  }
  if (stepIndex < activeStep || isComplete) {
    return <CheckCircleIcon sx={{ color: 'success.main' }} />;
  }
  if (stepIndex === activeStep) {
    return <CircularProgress size={20} />;
  }
  return undefined;
}

/* ------------------------------------------------------------------ */
/*  Log entry icon                                                    */
/* ------------------------------------------------------------------ */

function LogIcon({ icon }: { icon: LogEntry['icon'] }) {
  const sx16 = { fontSize: 16, flexShrink: 0, mt: '2px' } as const;
  switch (icon) {
    case 'check': return <CheckCircleIcon sx={{ ...sx16, color: 'success.main' }} />;
    case 'spinner': return <CircularProgress size={14} sx={{ flexShrink: 0 }} />;
    case 'error': return <ErrorOutlineIcon sx={{ ...sx16, color: 'error.main' }} />;
    case 'page': return <DescriptionOutlinedIcon sx={{ ...sx16, color: 'success.main' }} />;
    case 'search': return <SearchIcon sx={{ ...sx16, color: 'info.main' }} />;
    case 'write': return <EditNoteIcon sx={{ ...sx16, color: 'warning.main' }} />;
    default: return <InfoOutlinedIcon sx={{ ...sx16, color: 'text.secondary' }} />;
  }
}

/* ------------------------------------------------------------------ */
/*  Component                                                         */
/* ------------------------------------------------------------------ */

export function GenerationProgress({ events, onRetry }: GenerationProgressProps) {
  const logRef = useRef<HTMLDivElement>(null);

  const state = useMemo(() => deriveState(events), [events]);
  const log = useMemo(() => buildLog(events), [events]);
  const {
    activeStep,
    stepMessage,
    pagesCompleted,
    pagesTotal,
    isComplete,
    completionSummary,
    errorEvent,
    errorStep,
  } = state;

  // Auto-scroll log panel
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' });
  }, [log.length]);

  const header = isComplete
    ? 'Wiki generated!'
    : errorEvent
      ? errorEvent.type === 'task_cancelled'
        ? 'Generation cancelled'
        : 'Generation failed'
      : 'Generating wiki...';

  return (
    <Box sx={{ maxWidth: 900, mx: 'auto', mt: 4 }}>
      <Typography variant="h6" gutterBottom>
        {header}
      </Typography>

      {/* Two-panel layout: stepper left, log right */}
      <Box sx={{ display: 'flex', gap: 3, alignItems: 'flex-start' }}>
        {/* Left panel: Stepper */}
        <Box sx={{ width: 300, maxWidth: 300, minWidth: 0, flexShrink: 0, pb: 2 }}>
          <Stepper
            activeStep={isComplete ? STEPS.length : activeStep}
            orientation="vertical"
          >
            {STEPS.map((step, index) => {
              const isActive = index === activeStep && !isComplete && !errorEvent;
              const isGenerateStep = step.key === 'generate';

              return (
                <Step key={step.key} completed={index < activeStep || isComplete}>
                  <StepLabel
                    error={errorStep === index && !!errorEvent}
                    StepIconComponent={() => (
                      <StepIcon
                        stepIndex={index}
                        activeStep={activeStep}
                        errorStep={errorStep}
                        isComplete={isComplete}
                      />
                    )}
                    optional={
                      isActive && isGenerateStep && pagesTotal != null ? (
                        <Typography variant="caption" color="text.secondary">
                          {pagesCompleted} of {pagesTotal} sections
                        </Typography>
                      ) : undefined
                    }
                  >
                    {step.label}
                  </StepLabel>
                  <StepContent>
                    {isActive && stepMessage && (
                      <Typography variant="body2" color="text.secondary" noWrap title={stepMessage} sx={{ pb: 1 }}>
                        {stepMessage}
                      </Typography>
                    )}
                    {errorStep === index && errorEvent && (
                      <Typography variant="body2" color="error.main" sx={{ pb: 1, wordBreak: 'break-word' }}>
                        {errorEvent.type === 'task_cancelled'
                          ? (errorEvent as Extract<SSEEventData, { type: 'task_cancelled' }>).statusMessage ?? 'Generation cancelled'
                          : (errorEvent as Extract<SSEEventData, { type: 'error' | 'task_failed' }>).error}
                      </Typography>
                    )}
                  </StepContent>
                </Step>
              );
            })}
          </Stepper>

          {/* Error retry */}
          {errorEvent && onRetry && (
            <Button
              variant="contained"
              onClick={onRetry}
              sx={{
                mt: 2,
                ml: 4,
                '&:hover': { boxShadow: '0 0 24px rgba(255, 107, 74, 0.5)' },
              }}
            >
              Retry
            </Button>
          )}
        </Box>

        {/* Right panel: Activity log */}
        <Box
          ref={logRef}
          sx={{
            flex: 1,
            maxHeight: 400,
            minHeight: 200,
            overflow: 'auto',
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 2,
            bgcolor: 'background.paper',
          }}
        >
          <Box sx={{ px: 2, py: 1, borderBottom: '1px solid', borderColor: 'divider', position: 'sticky', top: 0, bgcolor: 'background.paper', zIndex: 1 }}>
            <Typography variant="caption" color="text.secondary" fontWeight={600} textTransform="uppercase" letterSpacing={0.5}>
              Activity
            </Typography>
          </Box>
          {log.map((entry, i) => (
            <Box
              key={i}
              sx={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 1,
                px: 2,
                py: 0.75,
                borderBottom: i < log.length - 1 ? '1px solid' : 'none',
                borderColor: 'divider',
                minWidth: 0,
              }}
            >
              <LogIcon icon={entry.icon} />
              <Typography
                variant="body2"
                sx={{
                  fontSize: '0.83rem',
                  color: entry.icon === 'error' ? 'error.main' : entry.icon === 'spinner' ? 'text.primary' : 'text.secondary',
                  fontWeight: entry.icon === 'spinner' ? 500 : 400,
                  wordBreak: 'break-word',
                  minWidth: 0,
                }}
              >
                {entry.text}
              </Typography>
            </Box>
          ))}
          {log.length === 0 && (
            <Box sx={{ px: 2, py: 3, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                Waiting for events...
              </Typography>
            </Box>
          )}
        </Box>
      </Box>

      {/* Completion summary */}
      {completionSummary && (
        <Alert severity="success" sx={{ mt: 2 }}>
          {completionSummary}
        </Alert>
      )}
    </Box>
  );
}
