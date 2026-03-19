import { useCallback, useState } from 'react';
import {
  Box,
  IconButton,
  InputAdornment,
  Menu,
  MenuItem,
  TextField,
  Typography,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import BoltIcon from '@mui/icons-material/Bolt';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';

interface AskBarProps {
  onSubmit: (question: string, deepResearch: boolean) => void;
  disabled?: boolean;
  repoLabel?: string;
}

type Mode = 'fast' | 'deep';

export function AskBar({ onSubmit, disabled = false, repoLabel }: AskBarProps) {
  const [input, setInput] = useState('');
  const [mode, setMode] = useState<Mode>('fast');
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const handleSubmit = useCallback(() => {
    const question = input.trim();
    if (!question || disabled) return;
    onSubmit(question, mode === 'deep');
    setInput('');
  }, [input, mode, disabled, onSubmit]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  const placeholder = repoLabel ? `Ask about ${repoLabel}...` : 'Ask about this repository...';

  return (
    <Box
      sx={{
        position: 'fixed',
        bottom: 24,
        left: '50%',
        transform: 'translateX(-50%)',
        width: { xs: '90%', sm: '75%', md: '65%', lg: '55%' },
        maxWidth: 700,
        zIndex: 1200,
      }}
    >
      <Box
        sx={(theme) => ({
          position: 'relative',
          bgcolor:
            theme.palette.mode === 'dark' ? 'rgba(17, 24, 39, 0.95)' : 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(16px)',
          borderRadius: 4,
          border: '1px solid transparent',
          backgroundClip: 'padding-box',
          boxShadow:
            theme.palette.mode === 'dark'
              ? '0 0 15px rgba(168, 85, 247, 0.4), 0 0 30px rgba(236, 72, 153, 0.2), 0 0 45px rgba(59, 130, 246, 0.15)'
              : '0 4px 24px rgba(0, 0, 0, 0.12), 0 0 15px rgba(168, 85, 247, 0.15)',
          px: 1.5,
          py: 1,
          '&::before': {
            content: '""',
            position: 'absolute',
            inset: -1,
            borderRadius: 'inherit',
            padding: '1px',
            background: 'linear-gradient(135deg, #ec4899, #a855f7, #3b82f6)',
            mask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
            maskComposite: 'exclude',
            WebkitMaskComposite: 'xor',
            pointerEvents: 'none',
          },
        })}
      >
        <TextField
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          fullWidth
          disabled={disabled}
          variant="standard"
          autoComplete="off"
          multiline
          maxRows={5}
          InputProps={{
            disableUnderline: true,
            startAdornment: (
              <InputAdornment position="start">
                <Box
                  onClick={(e) => setAnchorEl(e.currentTarget)}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0.25,
                    cursor: 'pointer',
                    px: 1,
                    py: 0.5,
                    borderRadius: 2,
                    bgcolor: 'action.hover',
                    '&:hover': { bgcolor: 'action.selected' },
                    transition: 'background-color 0.15s',
                  }}
                >
                  <BoltIcon sx={{ fontSize: 16, color: mode === 'deep' ? '#8B5CF6' : '#F59E0B' }} />
                  <Typography
                    variant="caption"
                    sx={{ fontWeight: 600, color: 'text.primary', fontSize: '0.72rem' }}
                  >
                    {mode === 'fast' ? 'Fast' : 'Deep'}
                  </Typography>
                  <KeyboardArrowDownIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
                </Box>
              </InputAdornment>
            ),
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  onClick={handleSubmit}
                  disabled={!input.trim() || disabled}
                  size="small"
                  sx={{
                    color: input.trim() ? 'primary.main' : 'text.secondary',
                    '&:hover': { color: 'primary.light' },
                  }}
                >
                  <SendIcon fontSize="small" />
                </IconButton>
              </InputAdornment>
            ),
            sx: {
              color: 'text.primary',
              fontSize: '0.9rem',
              py: 0.5,
            },
          }}
        />

        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={() => setAnchorEl(null)}
          anchorOrigin={{ vertical: 'top', horizontal: 'left' }}
          transformOrigin={{ vertical: 'bottom', horizontal: 'left' }}
          slotProps={{ paper: { sx: { minWidth: 140, mb: 1 } } }}
        >
          <MenuItem
            selected={mode === 'fast'}
            onClick={() => {
              setMode('fast');
              setAnchorEl(null);
            }}
          >
            <BoltIcon sx={{ fontSize: 16, color: '#F59E0B', mr: 1 }} />
            <Typography variant="body2">Fast</Typography>
          </MenuItem>
          <MenuItem
            selected={mode === 'deep'}
            onClick={() => {
              setMode('deep');
              setAnchorEl(null);
            }}
          >
            <BoltIcon sx={{ fontSize: 16, color: '#8B5CF6', mr: 1 }} />
            <Typography variant="body2">Deep Research</Typography>
          </MenuItem>
        </Menu>
      </Box>
    </Box>
  );
}
