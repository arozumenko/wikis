import { lazy, Suspense, useCallback, useRef, useState } from 'react';
import {
  Box,
  Chip,
  CircularProgress,
  Drawer,
  Fab,
  IconButton,
  Paper,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';
import ChatIcon from '@mui/icons-material/Chat';
import CloseIcon from '@mui/icons-material/Close';
import SendIcon from '@mui/icons-material/Send';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import BoltIcon from '@mui/icons-material/Bolt';
import SearchIcon from '@mui/icons-material/Search';
import { askQuestion, deepResearch } from '../api/wiki';
import { ThinkingSteps } from './ThinkingSteps';
import type { components } from '../api/types.generated';

const CodeMapTree = lazy(() => import('./CodeMapTree'));

type ChatMessage = components['schemas']['ChatMessage'];
type SourceReference = components['schemas']['SourceReference'];
type CodeMapData = components['schemas']['CodeMapData'];

type ChatMode = 'fast' | 'deep' | 'codemap';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceReference[];
  thinkingSteps?: string[];
  codeMap?: CodeMapData | null;
  mode?: ChatMode;
}

interface ChatDrawerProps {
  wikiId: string;
}

const DRAWER_WIDTH = 420;

export function ChatDrawer({ wikiId }: ChatDrawerProps) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<ChatMode>('fast');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    setTimeout(() => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);
  }, []);

  const handleSend = useCallback(async () => {
    const question = input.trim();
    if (!question || loading) return;

    setInput('');
    const userMsg: Message = { role: 'user', content: question, mode };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    scrollToBottom();

    try {
      const chatHistory: ChatMessage[] = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      let assistantMsg: Message;

      if (mode === 'fast') {
        const response = await askQuestion({
          wiki_id: wikiId,
          question,
          chat_history: chatHistory,
          k: 15,
        });
        assistantMsg = {
          role: 'assistant',
          content: response.answer,
          sources: response.sources,
          mode: 'fast',
        };
      } else {
        // 'deep' uses general research; 'codemap' uses the ask-first code map pipeline
        const response = await deepResearch({
          wiki_id: wikiId,
          question,
          research_type: mode === 'codemap' ? 'codemap' : 'general',
          enable_subagents: true,
        });
        assistantMsg = {
          role: 'assistant',
          content: response.answer,
          sources: response.sources,
          thinkingSteps: response.research_steps,
          codeMap: response.code_map,
          mode,
        };
      }
      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Sorry, something went wrong. Please try again.' },
      ]);
    } finally {
      setLoading(false);
      scrollToBottom();
    }
  }, [input, loading, messages, wikiId, mode, scrollToBottom]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <>
      {!open && (
        <Fab
          color="primary"
          onClick={() => setOpen(true)}
          sx={{ position: 'fixed', bottom: 24, right: 24 }}
          aria-label="Open chat"
        >
          <ChatIcon />
        </Fab>
      )}

      <Drawer
        anchor="right"
        open={open}
        onClose={() => setOpen(false)}
        variant="persistent"
        sx={{
          '& .MuiDrawer-paper': { width: DRAWER_WIDTH, p: 0 },
        }}
      >
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          {/* Header */}
          <Box
            sx={{
              p: 2,
              borderBottom: '1px solid',
              borderColor: 'divider',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <Typography variant="h6" sx={{ flex: 1 }}>
              Ask about this wiki
            </Typography>
            <IconButton onClick={() => setOpen(false)} size="small">
              <CloseIcon />
            </IconButton>
          </Box>

          {/* Messages */}
          <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
            {messages.length === 0 && (
              <Typography color="text.secondary" sx={{ textAlign: 'center', mt: 4 }}>
                Ask a question about this repository
              </Typography>
            )}

            {messages.map((msg, i) => {
              const isCodeMapMsg =
                msg.mode === 'codemap' &&
                msg.role === 'assistant' &&
                msg.codeMap != null &&
                (msg.codeMap.sections?.length ?? 0) >= 2;

              return (
                <Box key={i} sx={{ mb: 2 }}>
                  {/* For codemap mode: show graph as the primary content */}
                  {isCodeMapMsg ? (
                    <Box sx={{ mr: 4 }}>
                      <Suspense
                        fallback={
                          <CircularProgress size={24} sx={{ display: 'block', mx: 'auto', my: 2 }} />
                        }
                      >
                        <CodeMapTree data={msg.codeMap!} />
                      </Suspense>
                      {msg.content && (
                        <Paper
                          elevation={0}
                          sx={{ p: 1.5, mt: 1, bgcolor: 'action.hover', borderRadius: 2 }}
                        >
                          <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                            {msg.content}
                          </Typography>
                        </Paper>
                      )}
                    </Box>
                  ) : (
                    /* Fast / Deep: text answer bubble */
                    <Paper
                      elevation={0}
                      sx={{
                        p: 1.5,
                        bgcolor: msg.role === 'user' ? 'primary.main' : 'action.hover',
                        color: msg.role === 'user' ? 'primary.contrastText' : 'text.primary',
                        borderRadius: 2,
                        ml: msg.role === 'user' ? 4 : 0,
                        mr: msg.role === 'assistant' ? 4 : 0,
                      }}
                    >
                      <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                        {msg.content}
                      </Typography>
                    </Paper>
                  )}

                  {/* Thinking steps + sources for fast/deep (not codemap) */}
                  {!isCodeMapMsg && msg.role === 'assistant' && (
                    <Box sx={{ mr: 4 }}>
                      {msg.thinkingSteps && msg.thinkingSteps.length > 0 && (
                        <ThinkingSteps steps={msg.thinkingSteps} />
                      )}
                      {msg.sources && msg.sources.length > 0 && (
                        <Box sx={{ mt: 1, pl: 1 }}>
                          <Typography variant="caption" color="text.secondary">
                            Sources:
                          </Typography>
                          {msg.sources.map((src, j) => (
                            <Box
                              key={j}
                              sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 0.25, mt: 0.25 }}
                            >
                              {src.wiki_id && (
                                <Chip
                                  size="small"
                                  label={src.wiki_title ?? src.wiki_id}
                                  variant="outlined"
                                  color="primary"
                                  sx={{ fontSize: '0.6rem', height: 16 }}
                                />
                              )}
                              <Typography
                                variant="caption"
                                component="span"
                                sx={{ fontFamily: 'monospace', color: 'primary.main' }}
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
                  )}
                </Box>
              );
            })}

            {loading && (
              <Box sx={{ mb: 2 }}>
                <Paper
                  elevation={0}
                  sx={{ p: 1.5, bgcolor: 'action.hover', borderRadius: 2, mr: 4 }}
                >
                  <Typography variant="body2" color="text.secondary">
                    {mode === 'codemap' ? 'Building code map…' : mode === 'deep' ? 'Researching…' : 'Thinking…'}
                  </Typography>
                </Paper>
              </Box>
            )}

            <div ref={messagesEndRef} />
          </Box>

          {/* Input */}
          <Box sx={{ p: 2, borderTop: '1px solid', borderColor: 'divider' }}>
            {/* 3-way mode toggle */}
            <ToggleButtonGroup
              value={mode}
              exclusive
              onChange={(_, v) => { if (v) setMode(v); }}
              disabled={loading}
              size="small"
              fullWidth
              sx={{ mb: 1.5 }}
            >
              <ToggleButton value="fast" sx={{ py: 0.5, fontSize: 11, gap: 0.5 }}>
                <BoltIcon sx={{ fontSize: 14 }} /> Fast
              </ToggleButton>
              <ToggleButton value="deep" sx={{ py: 0.5, fontSize: 11, gap: 0.5 }}>
                <SearchIcon sx={{ fontSize: 14 }} /> Deep
              </ToggleButton>
              <ToggleButton value="codemap" sx={{ py: 0.5, fontSize: 11, gap: 0.5 }}>
                <AccountTreeIcon sx={{ fontSize: 14 }} /> Code Map
              </ToggleButton>
            </ToggleButtonGroup>

            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  mode === 'codemap'
                    ? 'Describe a flow to map…'
                    : mode === 'deep'
                    ? 'Ask a deep question…'
                    : 'Ask a question…'
                }
                size="small"
                fullWidth
                multiline
                maxRows={4}
                disabled={loading}
              />
              <IconButton onClick={handleSend} disabled={!input.trim() || loading} color="primary">
                <SendIcon />
              </IconButton>
            </Box>
          </Box>
        </Box>
      </Drawer>
    </>
  );
}
