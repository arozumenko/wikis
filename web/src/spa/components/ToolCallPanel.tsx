import { useEffect, useRef } from 'react';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Chip,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import BuildIcon from '@mui/icons-material/Build';
import type { ToolCallRecord, TodoItem } from '../api/sse';
import { TodoChecklist } from './TodoChecklist';

interface ToolCallPanelProps {
  toolCalls: ToolCallRecord[];
  todos?: TodoItem[];
}

const TODO_TOOLS = new Set(['write_todos', 'read_todos']);

function formatDuration(startIso: string, endIso: string): string {
  const ms = new Date(endIso).getTime() - new Date(startIso).getTime();
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function ToolCallPanel({ toolCalls, todos = [] }: ToolCallPanelProps) {
  const filteredCalls = toolCalls.filter((c) => !TODO_TOOLS.has(c.tool_name));
  const hasTodos = todos.length > 0;
  const hasToolCalls = filteredCalls.length > 0;
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevCountRef = useRef(0);

  // Auto-scroll to bottom when new tool calls appear
  useEffect(() => {
    if (filteredCalls.length > prevCountRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
    prevCountRef.current = filteredCalls.length;
  }, [filteredCalls.length]);

  if (!hasTodos && !hasToolCalls) {
    return (
      <Box sx={{ p: 3, color: 'text.secondary', textAlign: 'center' }}>
        <BuildIcon sx={{ fontSize: 28, mb: 1, opacity: 0.4 }} />
        <Typography variant="body2">No tool calls yet</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* Plan checklist (from todo_update events) — fixed at top */}
      {hasTodos && (
        <Box sx={{ flexShrink: 0 }}>
          <TodoChecklist todos={todos} />
        </Box>
      )}

      {/* Tool Calls — scrollable */}
      {hasToolCalls && (
        <Box ref={scrollRef} sx={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
          <Typography
            variant="caption"
            sx={{
              px: 2,
              pt: 2,
              pb: 1,
              display: 'block',
              fontWeight: 600,
              color: 'text.secondary',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              fontSize: '0.65rem',
              position: 'sticky',
              top: 0,
              bgcolor: 'background.default',
              zIndex: 1,
            }}
          >
            Tool Calls ({filteredCalls.length})
          </Typography>

          {filteredCalls.map((call, i) => (
            <Accordion
              key={i}
              defaultExpanded={false}
              disableGutters
              elevation={0}
              sx={{ '&:before': { display: 'none' }, bgcolor: 'transparent' }}
            >
              <AccordionSummary
                expandIcon={<ExpandMoreIcon sx={{ fontSize: 16 }} />}
                sx={{
                  minHeight: 36,
                  px: 2,
                  '& .MuiAccordionSummary-content': {
                    my: 0.5,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                  },
                }}
              >
                <Typography
                  variant="caption"
                  sx={{ fontFamily: 'monospace', fontSize: '0.75rem', fontWeight: 600, flex: 1 }}
                >
                  {call.tool_name}
                </Typography>
                {!call.done ? (
                  <Chip
                    label="running"
                    size="small"
                    color="info"
                    variant="outlined"
                    sx={{ height: 16, fontSize: '0.6rem' }}
                  />
                ) : (
                  call.endTimestamp && (
                    <Typography
                      variant="caption"
                      sx={{ color: 'text.secondary', fontSize: '0.6rem', fontFamily: 'monospace' }}
                    >
                      {formatDuration(call.timestamp, call.endTimestamp)}
                    </Typography>
                  )
                )}
              </AccordionSummary>

              <AccordionDetails sx={{ px: 2, pt: 0, pb: 1.5 }}>
                {/* Input */}
                <Typography
                  variant="caption"
                  sx={{
                    display: 'block',
                    color: 'text.secondary',
                    mb: 0.5,
                    textTransform: 'uppercase',
                    fontSize: '0.6rem',
                    letterSpacing: '0.05em',
                  }}
                >
                  Input
                </Typography>
                <Box
                  component="pre"
                  sx={{
                    m: 0,
                    mb: 1,
                    p: 1,
                    borderRadius: 1,
                    bgcolor: 'action.hover',
                    fontSize: '0.75rem',
                    lineHeight: 1.5,
                    maxHeight: 120,
                    overflowY: 'auto',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                  }}
                >
                  {call.tool_input}
                </Box>

                {/* Output */}
                {call.tool_output !== null && (
                  <>
                    <Typography
                      variant="caption"
                      sx={{
                        display: 'block',
                        color: 'text.secondary',
                        mb: 0.5,
                        textTransform: 'uppercase',
                        fontSize: '0.6rem',
                        letterSpacing: '0.05em',
                      }}
                    >
                      Output
                    </Typography>
                    <Box
                      component="pre"
                      sx={{
                        m: 0,
                        p: 1,
                        borderRadius: 1,
                        bgcolor: 'action.hover',
                        fontSize: '0.75rem',
                        lineHeight: 1.5,
                        maxHeight: 200,
                        overflowY: 'auto',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                      }}
                    >
                      {call.tool_output}
                    </Box>
                  </>
                )}
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      )}
    </Box>
  );
}
