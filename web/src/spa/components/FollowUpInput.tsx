import { useCallback, useState } from 'react';
import { Box, IconButton, TextField, Typography } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';

interface FollowUpInputProps {
  onSubmit: (question: string) => void;
  disabled?: boolean;
}

export function FollowUpInput({ onSubmit, disabled = false }: FollowUpInputProps) {
  const [input, setInput] = useState('');

  const handleSubmit = useCallback(() => {
    const question = input.trim();
    if (!question || disabled) return;
    onSubmit(question);
    setInput('');
  }, [input, disabled, onSubmit]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  return (
    <Box sx={{ mt: 4, pt: 3, borderTop: '1px solid', borderColor: 'divider' }}>
      <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
        Follow-up question
      </Typography>
      <Box sx={{ display: 'flex', gap: 1 }}>
        <TextField
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a follow-up..."
          size="small"
          fullWidth
          disabled={disabled}
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: 3,
              fontSize: '0.9rem',
            },
          }}
        />
        <IconButton
          onClick={handleSubmit}
          disabled={!input.trim() || disabled}
          color="primary"
          size="small"
        >
          <SendIcon />
        </IconButton>
      </Box>
    </Box>
  );
}
