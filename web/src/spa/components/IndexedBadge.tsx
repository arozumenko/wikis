import { Chip, Link, Tooltip, Typography, Box } from '@mui/material';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import { STALE_MS } from '../constants';

interface IndexedBadgeProps {
  indexedAt: string; // ISO 8601
  commitHash?: string;
  repoUrl?: string;
  provider?: string;
}

function relativeTime(iso: string): string {
  // Backend sends naive UTC timestamps (no Z suffix) — append Z so JS parses as UTC
  const utcIso = iso.endsWith('Z') || iso.includes('+') || iso.includes('-', 10) ? iso : iso + 'Z';
  const diff = Date.now() - new Date(utcIso).getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  return `${months}mo ago`;
}

function commitUrl(repoUrl: string, hash: string, provider: string): string {
  switch (provider) {
    case 'github':
      return `${repoUrl}/commit/${hash}`;
    case 'gitlab':
      return `${repoUrl}/-/commit/${hash}`;
    case 'bitbucket':
      return `${repoUrl}/commits/${hash}`;
    default:
      return `${repoUrl}/commit/${hash}`;
  }
}

export function IndexedBadge({
  indexedAt,
  commitHash,
  repoUrl,
  provider = 'github',
}: IndexedBadgeProps) {
  const isOutdated = Date.now() - new Date(indexedAt).getTime() > STALE_MS;
  const shortHash = commitHash?.slice(0, 7);
  const absolute = new Date(indexedAt).toLocaleString();

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <Tooltip title={`Indexed: ${absolute}`}>
        <Chip
          icon={isOutdated ? <WarningAmberIcon /> : <AccessTimeIcon />}
          label={relativeTime(indexedAt)}
          size="small"
          color={isOutdated ? 'warning' : 'default'}
          variant="outlined"
        />
      </Tooltip>

      {shortHash && (
        <Tooltip title={`Commit: ${commitHash}`}>
          {repoUrl ? (
            <Link
              href={commitUrl(repoUrl, commitHash!, provider)}
              target="_blank"
              rel="noopener"
              underline="hover"
              sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}
            >
              {shortHash}
            </Link>
          ) : (
            <Typography
              sx={{ fontFamily: 'monospace', fontSize: '0.75rem', color: 'text.secondary' }}
            >
              {shortHash}
            </Typography>
          )}
        </Tooltip>
      )}
    </Box>
  );
}
