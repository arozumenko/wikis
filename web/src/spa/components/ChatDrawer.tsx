import { useCallback, useRef, useState } from 'react';
import {
  Box,
  Drawer,
  Fab,
  FormControlLabel,
  IconButton,
  Paper,
  Switch,
  TextField,
  Typography,
} from '@mui/material';
import ChatIcon from '@mui/icons-material/Chat';
import CloseIcon from '@mui/icons-material/Close';
import SendIcon from '@mui/icons-material/Send';
import { askQuestion, deepResearch } from '../api/wiki';
import { ThinkingSteps } from './ThinkingSteps';
import type { components } from '../api/types.generated';

type ChatMessage = components['schemas']['ChatMessage'];
type SourceReference = components['schemas']['SourceReference'];

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceReference[];
  thinkingSteps?: string[];
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
  const [researchMode, setResearchMode] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    setTimeout(() => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);
  }, []);

  const handleSend = useCallback(async () => {
    const question = input.trim();
    if (!question || loading) return;

    setInput('');
    const userMsg: Message = { role: 'user', content: question };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    scrollToBottom();

    try {
      const chatHistory: ChatMessage[] = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      let assistantMsg: Message;

      if (researchMode) {
        const response = await deepResearch({
          wiki_id: wikiId,
          question,
          research_type: 'general',
          enable_subagents: true,
        });
        assistantMsg = {
          role: 'assistant',
          content: response.answer,
          sources: response.sources,
          thinkingSteps: response.research_steps,
        };
      } else {
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
  }, [input, loading, messages, wikiId, researchMode, scrollToBottom]);

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

            {messages.map((msg, i) => (
              <Box key={i} sx={{ mb: 2 }}>
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

                {msg.thinkingSteps && msg.thinkingSteps.length > 0 && (
                  <ThinkingSteps steps={msg.thinkingSteps} />
                )}

                {msg.sources && msg.sources.length > 0 && (
                  <Box sx={{ mt: 1, pl: 1 }}>
                    <Typography variant="caption" color="text.secondary">
                      Sources:
                    </Typography>
                    {msg.sources.map((src, j) => (
                      <Typography
                        key={j}
                        variant="caption"
                        component="div"
                        sx={{ fontFamily: 'monospace', color: 'primary.main' }}
                      >
                        {src.file_path}
                        {src.line_start ? `:${src.line_start}` : ''}
                        {src.line_end ? `-${src.line_end}` : ''}
                      </Typography>
                    ))}
                  </Box>
                )}
              </Box>
            ))}

            {loading && (
              <Box sx={{ mb: 2 }}>
                <Paper
                  elevation={0}
                  sx={{ p: 1.5, bgcolor: 'action.hover', borderRadius: 2, mr: 4 }}
                >
                  <Typography variant="body2" color="text.secondary">
                    Thinking...
                  </Typography>
                </Paper>
              </Box>
            )}

            <div ref={messagesEndRef} />
          </Box>

          {/* Input */}
          <Box sx={{ p: 2, borderTop: '1px solid', borderColor: 'divider' }}>
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={researchMode}
                  onChange={(e) => setResearchMode(e.target.checked)}
                  disabled={loading}
                  color="secondary"
                />
              }
              label={
                <Typography
                  variant="caption"
                  color={researchMode ? 'secondary.main' : 'text.secondary'}
                >
                  Deep Research
                </Typography>
              }
              sx={{ mb: 1, ml: 0 }}
            />
            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question..."
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
