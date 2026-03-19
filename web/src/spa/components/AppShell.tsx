import { useCallback, useState } from 'react';
import {
  AppBar,
  Avatar,
  Box,
  Chip,
  Divider,
  IconButton,
  Link as MuiLink,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  Toolbar,
  Tooltip,
  Typography,
} from '@mui/material';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import LogoutIcon from '@mui/icons-material/Logout';
import SettingsIcon from '@mui/icons-material/Settings';
import GitHubIcon from '@mui/icons-material/GitHub';
import RefreshIcon from '@mui/icons-material/Refresh';
import { Link, Outlet, useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { refreshWiki } from '../api/wiki';
import { ConfirmDialog } from './ConfirmDialog';
import { STALE_MS } from '../constants';

interface AppShellProps {
  mode: 'light' | 'dark';
  onToggleTheme: () => void;
  repoContext?: {
    wikiId?: string;
    repoUrl?: string;
    branch?: string;
    indexedAt?: string;
    commitHash?: string | null;
  };
}

function extractOwnerRepo(url: string): { owner: string; repo: string } | null {
  try {
    const parts = new URL(url).pathname.split('/').filter(Boolean);
    if (parts.length >= 2) return { owner: parts[0], repo: parts[1] };
  } catch {
    /* */
  }
  return null;
}

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return 'just now';
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export function AppShell({ mode, onToggleTheme, repoContext }: AppShellProps) {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const handleOpenMenu = useCallback((e: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(e.currentTarget);
  }, []);

  const handleCloseMenu = useCallback(() => {
    setAnchorEl(null);
  }, []);

  const [showRefreshConfirm, setShowRefreshConfirm] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const handleRefresh = useCallback(async () => {
    if (!repoContext?.wikiId) return;
    setShowRefreshConfirm(false);
    setRefreshing(true);
    try {
      const resp = await refreshWiki(repoContext.wikiId);
      // Reset before navigating — the stepper UI takes over from here
      setRefreshing(false);
      navigate(`/wiki/${repoContext.wikiId}?generating=true&invocation=${resp.invocation_id}`);
    } catch {
      setRefreshing(false);
    }
  }, [repoContext?.wikiId, navigate]);

  const parsed = repoContext?.repoUrl ? extractOwnerRepo(repoContext.repoUrl) : null;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <AppBar position="static" elevation={0}>
        <Toolbar
          variant="dense"
          sx={{ minHeight: 48, maxWidth: 1400, mx: 'auto', width: '100%', px: { xs: 2, md: 3 } }}
        >
          {/* Logo */}
          <Typography
            variant="h6"
            component={Link}
            to="/"
            sx={{
              fontWeight: 700,
              textDecoration: 'none',
              color: 'primary.main',
              mr: 1.5,
              letterSpacing: '-0.02em',
              fontSize: '1.1rem',
            }}
          >
            Wikis
          </Typography>

          {/* Repo context (inline when viewing a wiki) */}
          {parsed && (
            <>
              <Typography variant="body2" color="text.secondary" sx={{ mx: 0.5 }}>
                /
              </Typography>
              <MuiLink
                href={repoContext!.repoUrl}
                target="_blank"
                rel="noopener"
                underline="hover"
                color="text.secondary"
                variant="body2"
              >
                {parsed.owner}
              </MuiLink>
              <Typography variant="body2" color="text.secondary" sx={{ mx: 0.25 }}>
                /
              </Typography>
              <MuiLink
                href={repoContext!.repoUrl}
                target="_blank"
                rel="noopener"
                underline="hover"
                color="text.primary"
                variant="body2"
                sx={{ fontWeight: 600 }}
              >
                {parsed.repo}
              </MuiLink>

              <Tooltip title="View on GitHub">
                <IconButton
                  component="a"
                  href={repoContext!.repoUrl}
                  target="_blank"
                  rel="noopener"
                  size="small"
                  sx={{ color: 'text.secondary', ml: 0.5 }}
                >
                  <GitHubIcon sx={{ fontSize: 16 }} />
                </IconButton>
              </Tooltip>

              {repoContext!.branch && (
                <Chip
                  label={repoContext!.branch}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: '0.65rem', height: 20, ml: 0.5 }}
                />
              )}

              {repoContext!.commitHash && (
                <Tooltip title={`Commit: ${repoContext!.commitHash}`}>
                  <Chip
                    label={repoContext!.commitHash.slice(0, 7)}
                    size="small"
                    variant="outlined"
                    component="a"
                    href={`${repoContext!.repoUrl}/commit/${repoContext!.commitHash}`}
                    target="_blank"
                    rel="noopener"
                    clickable
                    sx={{ fontSize: '0.65rem', height: 20, ml: 0.5, fontFamily: 'monospace' }}
                  />
                </Tooltip>
              )}

              {repoContext!.indexedAt &&
                Date.now() - new Date(repoContext!.indexedAt).getTime() > STALE_MS && (
                  <Tooltip title="Wiki was indexed more than 30 days ago and may be outdated">
                    <Chip
                      label="Stale"
                      size="small"
                      color="warning"
                      sx={{ fontSize: '0.65rem', height: 20, ml: 0.5 }}
                    />
                  </Tooltip>
                )}
            </>
          )}

          <Box sx={{ flexGrow: 1 }} />

          {/* Refresh button (right side) */}
          {repoContext?.wikiId && (
            <Tooltip title="Refresh wiki">
              <IconButton
                size="small"
                onClick={() => setShowRefreshConfirm(true)}
                disabled={refreshing}
                sx={{ color: 'text.secondary', mr: 0.5, '&:hover': { color: 'primary.main' } }}
              >
                <RefreshIcon
                  sx={{
                    fontSize: 16,
                    ...(refreshing && {
                      animation: 'spin 1s linear infinite',
                      '@keyframes spin': {
                        from: { transform: 'rotate(0deg)' },
                        to: { transform: 'rotate(360deg)' },
                      },
                    }),
                  }}
                />
              </IconButton>
            </Tooltip>
          )}

          {/* Indexed timestamp */}
          {repoContext?.indexedAt && (
            <Tooltip title={`Indexed: ${new Date(repoContext.indexedAt).toLocaleString()}`}>
              <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
                {relativeTime(repoContext.indexedAt)}
              </Typography>
            </Tooltip>
          )}

          {/* Theme toggle */}
          <IconButton
            onClick={onToggleTheme}
            sx={{
              color: 'text.secondary',
              transition: 'color 0.2s, transform 0.2s',
              '&:hover': { color: 'text.primary', transform: 'rotate(20deg)' },
            }}
            aria-label="Toggle theme"
            size="small"
          >
            {mode === 'dark' ? (
              <LightModeIcon fontSize="small" />
            ) : (
              <DarkModeIcon fontSize="small" />
            )}
          </IconButton>

          {/* User menu */}
          {user && (
            <>
              <IconButton onClick={handleOpenMenu} sx={{ ml: 0.5, p: 0.25 }}>
                <Avatar
                  src={user.image ?? undefined}
                  alt={user.name ?? ''}
                  sx={{ width: 28, height: 28 }}
                />
              </IconButton>

              <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleCloseMenu}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                transformOrigin={{ vertical: 'top', horizontal: 'right' }}
                slotProps={{ paper: { sx: { minWidth: 180, mt: 1 } } }}
              >
                <Box sx={{ px: 2, py: 1 }}>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {user.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {user.email}
                  </Typography>
                </Box>
                <Divider />
                <MenuItem
                  onClick={() => {
                    handleCloseMenu();
                    navigate('/settings');
                  }}
                >
                  <ListItemIcon>
                    <SettingsIcon fontSize="small" />
                  </ListItemIcon>
                  <ListItemText>Settings</ListItemText>
                </MenuItem>
                <MenuItem
                  onClick={() => {
                    handleCloseMenu();
                    signOut();
                  }}
                >
                  <ListItemIcon>
                    <LogoutIcon fontSize="small" sx={{ color: 'error.main' }} />
                  </ListItemIcon>
                  <ListItemText sx={{ color: 'error.main' }}>Sign out</ListItemText>
                </MenuItem>
              </Menu>
            </>
          )}
        </Toolbar>
      </AppBar>

      <Box
        component="main"
        sx={{
          flex: 1,
          '@keyframes fadeInUp': {
            from: { opacity: 0, transform: 'translateY(8px)' },
            to: { opacity: 1, transform: 'translateY(0)' },
          },
          '& > *': { animation: 'fadeInUp 0.3s ease-out' },
        }}
      >
        <Outlet />
      </Box>

      <ConfirmDialog
        open={showRefreshConfirm}
        title="Refresh Wiki"
        message="Re-index the repository and regenerate all wiki pages? This may take a few minutes."
        confirmLabel="Refresh"
        onConfirm={handleRefresh}
        onCancel={() => setShowRefreshConfirm(false)}
      />
    </Box>
  );
}
