import { Box, Chip, IconButton, Typography } from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { CopyAnswerButton } from './CopyAnswerButton';

import AccountTreeIcon from '@mui/icons-material/AccountTree';

interface AnswerHeaderProps {
  question: string;
  mode: 'fast' | 'deep' | 'codemap';
  onBack: () => void;
  answer?: string | null;
  turnCount?: number;
}

export function AnswerHeader({ question, mode, onBack, answer, turnCount }: AnswerHeaderProps) {
  const title = (turnCount ?? 1) > 1 ? `Conversation (${turnCount} turns)` : question;
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
        {title}
      </Typography>
      {mode === 'codemap' ? (
        <Chip
          icon={<AccountTreeIcon sx={{ fontSize: '0.75rem !important' }} />}
          label="Code Map"
          size="small"
          color="success"
          variant="outlined"
          sx={{ fontSize: '0.65rem', height: 22 }}
        />
      ) : (
        <Chip
          label={mode === 'deep' ? 'Deep Research' : 'Fast'}
          size="small"
          color={mode === 'deep' ? 'secondary' : 'default'}
          variant="outlined"
          sx={{ fontSize: '0.65rem', height: 22 }}
        />
      )}
      {answer && <CopyAnswerButton text={answer} />}
    </Box>
  );
}
