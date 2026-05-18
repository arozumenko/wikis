/**
 * ConnectionsPage — lists stored connections and provides an "Add Connection"
 * button to launch the AddConnectionWizard.
 *
 * This is rendered as the "Connections" tab inside SettingsPage.
 */
import { useState } from 'react';
import {
  Box,
  Button,
  Typography,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import { useConnections } from '../hooks/useConnections';
import {
  AtlassianConnectionCard,
  GitConnectionCard,
} from '../components/connections/ConnectionCard';
import { AddConnectionWizard } from '../components/connections/AddConnectionWizard';

export function ConnectionsPage() {
  const {
    connections,
    atlassian,
    removeAtlassian,
    removeGitConnection,
    refreshAtlassianIfNeeded,
  } = useConnections();

  const [wizardOpen, setWizardOpen] = useState(false);

  const gitConnections = connections.filter((c) => c.provider === 'git');

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="h6">Connections</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setWizardOpen(true)}
          size="small"
        >
          Add Connection
        </Button>
      </Box>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Source credentials stored in this browser. Used when generating wikis from
        private Atlassian (Confluence / Jira) or Git repositories.
      </Typography>

      {connections.length === 0 && (
        <Box
          sx={{
            border: '1px dashed',
            borderColor: 'divider',
            borderRadius: 2,
            py: 6,
            textAlign: 'center',
          }}
        >
          <Typography color="text.secondary">
            No connections yet. Click "Add Connection" to get started.
          </Typography>
        </Box>
      )}

      {atlassian && (
        <Box sx={{ mb: 2 }}>
          <AtlassianConnectionCard
            connection={atlassian}
            onRemove={removeAtlassian}
            onRefresh={refreshAtlassianIfNeeded}
          />
        </Box>
      )}

      {gitConnections.map((c) => {
        // Safe cast — we filtered above
        const git = c as typeof c & { provider: 'git' };
        return (
          <Box key={git.id} sx={{ mb: 2 }}>
            <GitConnectionCard
              connection={git}
              onRemove={() => removeGitConnection(git.id)}
            />
          </Box>
        );
      })}

      <AddConnectionWizard open={wizardOpen} onClose={() => setWizardOpen(false)} />
    </Box>
  );
}
