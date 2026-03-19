import { Box, Typography } from '@mui/material';

export function ThinkingIndicator() {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, py: 4 }}>
      <Box
        sx={{
          display: 'flex',
          gap: 0.5,
          '& span': {
            width: 6,
            height: 6,
            borderRadius: '50%',
            bgcolor: 'primary.main',
            animation: 'bounce 1.4s infinite ease-in-out both',
          },
          '& span:nth-of-type(1)': { animationDelay: '-0.32s' },
          '& span:nth-of-type(2)': { animationDelay: '-0.16s' },
          '@keyframes bounce': {
            '0%, 80%, 100%': { transform: 'scale(0)' },
            '40%': { transform: 'scale(1)' },
          },
        }}
      >
        <span />
        <span />
        <span />
      </Box>
      <Typography variant="body2" color="text.secondary">
        Thinking...
      </Typography>
    </Box>
  );
}
