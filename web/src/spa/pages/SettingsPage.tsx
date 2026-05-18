import { useCallback, useEffect, useState } from 'react';
import {
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  Paper,
  Snackbar,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  TextField,
  Typography,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import { IntegrationsTab } from '../components/IntegrationsTab';
import { AccountTab } from '../components/AccountTab';

const AUTH_URL = ''; // Same origin

interface ApiKeyEntry {
  id: string;
  name: string | null;
  start: string | null;
  createdAt: string;
  expiresAt: string | null;
  lastRequest: string | null;
  enabled: boolean;
}

export function SettingsPage() {
  const [tab, setTab] = useState(0);
  const [keys, setKeys] = useState<ApiKeyEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [snackbar, setSnackbar] = useState('');

  const fetchKeys = useCallback(async () => {
    try {
      const resp = await fetch(`${AUTH_URL}/api/auth/api-key/list`, {
        credentials: 'include',
      });
      if (resp.ok) {
        const data = await resp.json();
        setKeys(data.apiKeys ?? []);
      }
    } catch {
      // Silent fail
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchKeys();
  }, [fetchKeys]);

  const handleCreate = useCallback(async () => {
    try {
      const resp = await fetch(`${AUTH_URL}/api/auth/api-key/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ name: newKeyName || undefined }),
      });
      if (resp.ok) {
        const data = await resp.json();
        setCreatedKey(data.key);
        setNewKeyName('');
        setShowCreate(false);
        fetchKeys();
      }
    } catch {
      setSnackbar('Failed to create API key');
    }
  }, [newKeyName, fetchKeys]);

  const handleRevoke = useCallback(async (keyId: string) => {
    try {
      await fetch(`${AUTH_URL}/api/auth/api-key/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ keyId }),
      });
      setKeys((prev) => prev.filter((k) => k.id !== keyId));
      setSnackbar('API key revoked');
    } catch {
      setSnackbar('Failed to revoke key');
    }
  }, []);

  const handleCopy = useCallback((text: string) => {
    navigator.clipboard.writeText(text);
    setSnackbar('Copied to clipboard');
  }, []);

  return (
    <Box sx={{ px: { xs: 3, sm: 4, md: 6, lg: 8 }, py: 4, maxWidth: 900, mx: 'auto' }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Settings
      </Typography>

      <Tabs
        value={tab}
        onChange={(_, v) => setTab(v)}
        sx={{
          mb: 4,
          borderBottom: '1px solid',
          borderColor: 'divider',
          '& .MuiTab-root': { textTransform: 'none', fontSize: '0.95rem' },
        }}
      >
        <Tab label="Account" />
        <Tab label="API Keys" />
        <Tab label="Integrations" />
      </Tabs>

      {/* Account tab */}
      {tab === 0 && <AccountTab />}

      {/* API Keys tab */}
      {tab === 1 && (
        <Box>
          <Typography variant="h6" sx={{ mb: 1 }}>
            API Keys
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Personal access tokens for API authentication. Tokens are prefixed with{' '}
            <code>wikis_</code>.
          </Typography>

          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setShowCreate(true)}
            sx={{ mb: 3 }}
          >
            Generate New Key
          </Button>

          {!loading && keys.length === 0 && (
            <Typography color="text.secondary">No API keys yet.</Typography>
          )}

          {keys.length > 0 && (
            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Key</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell>Expires</TableCell>
                    <TableCell>Last Used</TableCell>
                    <TableCell />
                  </TableRow>
                </TableHead>
                <TableBody>
                  {keys.map((key) => (
                    <TableRow key={key.id}>
                      <TableCell>{key.name ?? '—'}</TableCell>
                      <TableCell>
                        <code>{key.start ?? 'wikis_'}•••</code>
                      </TableCell>
                      <TableCell>{new Date(key.createdAt).toLocaleDateString()}</TableCell>
                      <TableCell>
                        {key.expiresAt ? (
                          new Date(key.expiresAt) < new Date() ? (
                            <Chip label="Expired" size="small" color="error" />
                          ) : (
                            new Date(key.expiresAt).toLocaleDateString()
                          )
                        ) : (
                          'Never'
                        )}
                      </TableCell>
                      <TableCell>
                        {key.lastRequest
                          ? new Date(key.lastRequest).toLocaleDateString()
                          : 'Never'}
                      </TableCell>
                      <TableCell>
                        <IconButton
                          size="small"
                          onClick={() => handleRevoke(key.id)}
                          color="error"
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Box>
      )}

      {/* Integrations tab */}
      {tab === 2 && <IntegrationsTab />}

      {/* Create dialog */}
      <Dialog open={showCreate} onClose={() => setShowCreate(false)}>
        <DialogTitle>Generate API Key</DialogTitle>
        <DialogContent>
          <TextField
            label="Key name (optional)"
            value={newKeyName}
            onChange={(e) => setNewKeyName(e.target.value)}
            fullWidth
            margin="normal"
            placeholder="e.g., CI/CD Pipeline"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowCreate(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleCreate}>
            Generate
          </Button>
        </DialogActions>
      </Dialog>

      {/* Show created key (once) */}
      <Dialog
        open={createdKey !== null}
        onClose={() => setCreatedKey(null)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>API Key Created</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Copy this key now — it won't be shown again.
          </Typography>
          <Box
            sx={{
              p: 2,
              bgcolor: 'action.hover',
              borderRadius: 1,
              fontFamily: 'monospace',
              fontSize: '0.85rem',
              wordBreak: 'break-all',
              display: 'flex',
              alignItems: 'center',
              gap: 1,
            }}
          >
            <code style={{ flex: 1 }}>{createdKey}</code>
            <IconButton size="small" onClick={() => handleCopy(createdKey!)}>
              <ContentCopyIcon fontSize="small" />
            </IconButton>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button variant="contained" onClick={() => setCreatedKey(null)}>
            Done
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={!!snackbar}
        autoHideDuration={3000}
        onClose={() => setSnackbar('')}
        message={snackbar}
      />
    </Box>
  );
}
