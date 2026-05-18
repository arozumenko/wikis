/**
 * ConnectionCard — MUI card representing a single stored connection
 * (Atlassian or Git PAT) with Test / Refresh / Delete actions.
 */
import { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Typography,
} from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import { ConfirmDialog } from '../ConfirmDialog';
import { fetchAccessibleResources } from '../../lib/atlassian-oauth';
import type { AtlassianConnection, GitConnection } from '../../hooks/useConnections';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AtlassianCardProps {
  connection: AtlassianConnection;
  onRemove: () => void;
  onRefresh: () => Promise<AtlassianConnection | null>;
}

interface GitCardProps {
  connection: GitConnection;
  onRemove: () => void;
}

// ---------------------------------------------------------------------------
// Atlassian card
// ---------------------------------------------------------------------------

export function AtlassianConnectionCard({
  connection,
  onRemove,
  onRefresh,
}: AtlassianCardProps) {
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<'ok' | 'fail' | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshResult, setRefreshResult] = useState<'ok' | 'fail' | null>(null);
  const [confirmOpen, setConfirmOpen] = useState(false);

  async function handleTest() {
    setTesting(true);
    setTestResult(null);
    try {
      // Try to fetch accessible resources — if the token is valid we get data back
      await fetchAccessibleResources(connection.access_token);
      setTestResult('ok');
    } catch {
      setTestResult('fail');
    } finally {
      setTesting(false);
    }
  }

  async function handleRefresh() {
    setRefreshing(true);
    setRefreshResult(null);
    try {
      await onRefresh();
      setRefreshResult('ok');
      // Auto-clear the success message after 3 seconds
      setTimeout(() => setRefreshResult(null), 3000);
    } catch {
      setRefreshResult('fail');
    } finally {
      setRefreshing(false);
    }
  }

  const expiresLabel = connection.expires_at
    ? new Date(connection.expires_at).toLocaleString()
    : 'Unknown';

  const isExpired = connection.expires_at < Date.now();

  return (
    <>
      <Card variant="outlined" sx={{ borderRadius: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5 }}>
            {/* Provider logo placeholder — Atlassian blue square */}
            <Box
              sx={{
                width: 36,
                height: 36,
                borderRadius: 1,
                bgcolor: '#0052CC',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
                color: '#fff',
                fontWeight: 700,
                fontSize: '0.75rem',
                letterSpacing: '-0.02em',
              }}
            >
              AT
            </Box>

            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography variant="subtitle2" noWrap>
                Connected to {connection.site_name || 'Atlassian'}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Cloud ID: {connection.cloud_id}
              </Typography>
              <Box sx={{ mt: 0.5, display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
                {isExpired ? (
                  <Chip label="Expired" size="small" color="error" variant="outlined" />
                ) : (
                  <Chip label="Active" size="small" color="success" variant="outlined" />
                )}
                <Typography variant="caption" color="text.secondary">
                  Expires {expiresLabel}
                </Typography>
                {testResult === 'ok' && (
                  <CheckCircleOutlineIcon
                    sx={{ fontSize: 16, color: 'success.main' }}
                  />
                )}
                {testResult === 'fail' && (
                  <ErrorOutlineIcon sx={{ fontSize: 16, color: 'error.main' }} />
                )}
              </Box>
            </Box>
          </Box>

          <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
            <Button
              size="small"
              variant="outlined"
              disabled={testing}
              onClick={handleTest}
              startIcon={testing ? <CircularProgress size={14} /> : undefined}
            >
              {testing ? 'Testing…' : 'Test'}
            </Button>
            <Button
              size="small"
              variant="outlined"
              disabled={refreshing}
              onClick={handleRefresh}
              startIcon={refreshing ? <CircularProgress size={14} /> : undefined}
            >
              {refreshing ? 'Refreshing…' : 'Refresh'}
            </Button>
            {refreshResult === 'ok' && (
              <Typography variant="caption" color="success.main">
                Token refreshed
              </Typography>
            )}
            {refreshResult === 'fail' && (
              <Typography variant="caption" color="error.main">
                Refresh failed — check your connection
              </Typography>
            )}
            <Button
              size="small"
              variant="outlined"
              color="error"
              onClick={() => setConfirmOpen(true)}
            >
              Delete
            </Button>
          </Box>

          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1.5 }}>
            Added {new Date(connection.created_at).toLocaleDateString()}
          </Typography>
        </CardContent>
      </Card>

      <ConfirmDialog
        open={confirmOpen}
        title="Remove Atlassian connection"
        message="This will remove the stored tokens. You will need to reconnect to use Atlassian sources in future wiki generations."
        confirmLabel="Remove"
        onConfirm={() => {
          setConfirmOpen(false);
          onRemove();
        }}
        onCancel={() => setConfirmOpen(false)}
      />
    </>
  );
}

// ---------------------------------------------------------------------------
// Git PAT card
// ---------------------------------------------------------------------------

export function GitConnectionCard({ connection, onRemove }: GitCardProps) {
  const [confirmOpen, setConfirmOpen] = useState(false);
  const isValid = !!connection.pat;

  return (
    <>
      <Card variant="outlined" sx={{ borderRadius: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5 }}>
            {/* Provider logo placeholder */}
            <Box
              sx={{
                width: 36,
                height: 36,
                borderRadius: 1,
                bgcolor: 'action.selected',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
                fontWeight: 700,
                fontSize: '0.75rem',
                color: 'text.secondary',
              }}
            >
              GIT
            </Box>

            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography variant="subtitle2" noWrap>
                {connection.label || connection.repo_url}
              </Typography>
              {connection.label && (
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ display: 'block' }}
                  noWrap
                >
                  {connection.repo_url}
                </Typography>
              )}
              <Box sx={{ mt: 0.5, display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
                <Chip label={connection.branch} size="small" variant="outlined" />
                {isValid ? (
                  <CheckCircleOutlineIcon sx={{ fontSize: 16, color: 'success.main' }} />
                ) : (
                  <ErrorOutlineIcon sx={{ fontSize: 16, color: 'error.main' }} />
                )}
                <Typography variant="caption" color="text.secondary">
                  {isValid ? 'PAT stored' : 'No PAT — edit to add one'}
                </Typography>
              </Box>
            </Box>
          </Box>

          <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
            <Button
              size="small"
              variant="outlined"
              color="error"
              onClick={() => setConfirmOpen(true)}
            >
              Delete
            </Button>
          </Box>

          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1.5 }}>
            Added {new Date(connection.created_at).toLocaleDateString()}
          </Typography>
        </CardContent>
      </Card>

      <ConfirmDialog
        open={confirmOpen}
        title="Remove Git connection"
        message={`Remove the stored PAT for "${connection.label || connection.repo_url}"? You can re-add it at any time.`}
        confirmLabel="Remove"
        onConfirm={() => {
          setConfirmOpen(false);
          onRemove();
        }}
        onCancel={() => setConfirmOpen(false)}
      />
    </>
  );
}
