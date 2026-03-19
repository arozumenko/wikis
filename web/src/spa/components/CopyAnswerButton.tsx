import { IconButton, Snackbar, Tooltip } from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import { useCopyToClipboard } from '../hooks/useCopyToClipboard';

interface CopyAnswerButtonProps {
  text: string;
}

export function CopyAnswerButton({ text }: CopyAnswerButtonProps) {
  const { copied, copy } = useCopyToClipboard();

  return (
    <>
      <Tooltip title={copied ? 'Copied!' : 'Copy answer'}>
        <IconButton
          onClick={() => copy(text)}
          size="small"
          sx={{ opacity: 0.6, '&:hover': { opacity: 1 } }}
        >
          <ContentCopyIcon sx={{ fontSize: 14 }} />
        </IconButton>
      </Tooltip>
      <Snackbar
        open={copied}
        autoHideDuration={2000}
        message="Answer copied"
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      />
    </>
  );
}
