import { Box, Chip, IconButton, Link, Tooltip, Typography } from '@mui/material';
import GitHubIcon from '@mui/icons-material/GitHub';
import { STALE_MS } from '../constants';

interface RepoHeaderProps {
  repoUrl?: string;
  branch?: string;
  indexedAt?: string;
  commitHash?: string | null;
}

function extractOwnerRepo(url: string): { owner: string; repo: string } | null {
  try {
    const parts = new URL(url).pathname.split('/').filter(Boolean);
    if (parts.length >= 2) return { owner: parts[0], repo: parts[1] };
  } catch {
    // Not a valid URL
  }
  return null;
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
  return `${days}d ago`;
}

export function RepoHeader({ repoUrl, branch, indexedAt, commitHash }: RepoHeaderProps) {
  const parsed = repoUrl ? extractOwnerRepo(repoUrl) : null;

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1.5,
        px: 2,
        py: 1.5,
        borderBottom: '1px solid',
        borderColor: 'divider',
      }}
    >
      {parsed ? (
        <Typography variant="body2" sx={{ fontWeight: 500 }}>
          <Link
            href={repoUrl}
            target="_blank"
            rel="noopener"
            underline="hover"
            color="text.secondary"
          >
            {parsed.owner}
          </Link>
          {' / '}
          <Link
            href={repoUrl}
            target="_blank"
            rel="noopener"
            underline="hover"
            color="text.primary"
            sx={{ fontWeight: 600 }}
          >
            {parsed.repo}
          </Link>
        </Typography>
      ) : (
        <Typography variant="body2" color="text.secondary">
          Wiki
        </Typography>
      )}

      {repoUrl && (
        <Tooltip title="View on GitHub">
          <IconButton
            component="a"
            href={repoUrl}
            target="_blank"
            rel="noopener"
            size="small"
            sx={{ color: 'text.secondary' }}
          >
            <GitHubIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      )}

      {branch && (
        <Chip
          label={branch}
          size="small"
          variant="outlined"
          sx={{ fontSize: '0.7rem', height: 22 }}
        />
      )}

      <Box sx={{ flexGrow: 1 }} />

      {commitHash && (
        <Tooltip title={`Commit: ${commitHash}`}>
          <Chip
            label={commitHash.slice(0, 7)}
            size="small"
            variant="outlined"
            component="a"
            href={repoUrl ? `${repoUrl}/commit/${commitHash}` : undefined}
            target="_blank"
            rel="noopener"
            clickable={!!repoUrl}
            sx={{ fontSize: '0.65rem', height: 20, fontFamily: 'monospace' }}
          />
        </Tooltip>
      )}

      {indexedAt && Date.now() - new Date(indexedAt.endsWith('Z') || indexedAt.includes('+') || indexedAt.includes('-', 10) ? indexedAt : indexedAt + 'Z').getTime() > STALE_MS && (
        <Tooltip title="Wiki was indexed more than 30 days ago and may be outdated">
          <Chip
            label="Stale"
            size="small"
            color="warning"
            sx={{ fontSize: '0.65rem', height: 20 }}
          />
        </Tooltip>
      )}

      {indexedAt && (
        <Tooltip title={`Indexed: ${new Date(indexedAt).toLocaleString()}`}>
          <Typography variant="caption" color="text.secondary">
            {relativeTime(indexedAt)}
          </Typography>
        </Tooltip>
      )}
    </Box>
  );
}
