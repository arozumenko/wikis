import { Box, Chip, IconButton, Typography } from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { CopyAnswerButton } from './CopyAnswerButton';

interface AnswerHeaderProps {
  question: string;
  mode: 'fast' | 'deep';
  onBack: () => void;
  answer?: string | null;
}

export function AnswerHeader({ question, mode, onBack, answer }: AnswerHeaderProps) {
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1.5,
        px: 3,
        py: 1.5,
        borderBottom: '1px solid',
        borderColor: 'divider',
      }}
    >
      <IconButton onClick={onBack} size="small" sx={{ color: 'text.secondary' }}>
        <ArrowBackIcon fontSize="small" />
      </IconButton>
      <Typography
        variant="body1"
        sx={{
          flex: 1,
          fontWeight: 500,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {question}
      </Typography>
      <Chip
        label={mode === 'deep' ? 'Deep Research' : 'Fast'}
        size="small"
        color={mode === 'deep' ? 'secondary' : 'default'}
        variant="outlined"
        sx={{ fontSize: '0.65rem', height: 22 }}
      />
      {answer && <CopyAnswerButton text={answer} />}
    </Box>
  );
}
