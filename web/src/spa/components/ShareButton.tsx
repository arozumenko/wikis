import { IconButton, Snackbar, Tooltip } from '@mui/material';
import ShareIcon from '@mui/icons-material/Share';
import { useCopyToClipboard } from '../hooks/useCopyToClipboard';

interface ShareButtonProps {
  /** URL to copy. Defaults to current page URL. */
  url?: string;
}

export function ShareButton({ url }: ShareButtonProps) {
  const { copied, copy } = useCopyToClipboard();

  const handleClick = () => {
    copy(url ?? window.location.href);
  };

  return (
    <>
      <Tooltip title="Share link">
        <IconButton onClick={handleClick} size="small">
          <ShareIcon fontSize="small" />
        </IconButton>
      </Tooltip>
      <Snackbar
        open={copied}
        autoHideDuration={2000}
        message="Link copied"
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      />
    </>
  );
}
