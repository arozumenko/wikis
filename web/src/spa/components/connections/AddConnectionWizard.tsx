/**
 * AddConnectionWizard — 3-step MUI Dialog for adding a new connection.
 *
 * Step 1: Pick connector (Atlassian or Git)
 * Step 2: Configure (Atlassian: OAuth popup; Git: PAT form)
 * Step 3: Confirm & save
 */
import { useRef, useState } from 'react';
import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Paper,
  Step,
  StepLabel,
  Stepper,
  Typography,
} from '@mui/material';
import { AtlassianStep } from './AtlassianStep';
import { GitPATStep } from './GitPATStep';
import { useConnections } from '../../hooks/useConnections';
import type { AtlassianConnection, GitConnection } from '../../hooks/useConnections';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ConnectorType = 'atlassian' | 'git';

const STEPS = ['Choose connector', 'Configure', 'Confirm'] as const;

interface AddConnectionWizardProps {
  open: boolean;
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// Connector tile (step 1)
// ---------------------------------------------------------------------------

interface ConnectorTileProps {
  selected: boolean;
  onClick: () => void;
  logo: React.ReactNode;
  label: string;
  description: string;
}

function ConnectorTile({ selected, onClick, logo, label, description }: ConnectorTileProps) {
  return (
    <Paper
      variant="outlined"
      onClick={onClick}
      sx={{
        p: 2.5,
        cursor: 'pointer',
        borderRadius: 2,
        border: '2px solid',
        borderColor: selected ? 'primary.main' : 'divider',
        bgcolor: selected ? 'action.selected' : 'transparent',
        transition: 'border-color 0.15s, background-color 0.15s',
        '&:hover': {
          borderColor: selected ? 'primary.main' : 'action.hover',
          bgcolor: 'action.hover',
        },
        display: 'flex',
        alignItems: 'center',
        gap: 2,
      }}
    >
      {logo}
      <Box>
        <Typography variant="subtitle2">{label}</Typography>
        <Typography variant="caption" color="text.secondary">
          {description}
        </Typography>
      </Box>
    </Paper>
  );
}

// ---------------------------------------------------------------------------
// Wizard
// ---------------------------------------------------------------------------

export function AddConnectionWizard({ open, onClose }: AddConnectionWizardProps) {
  const [activeStep, setActiveStep] = useState(0);
  const [connector, setConnector] = useState<ConnectorType | null>(null);
  // Pending Atlassian connection (set after OAuth success, before user confirms)
  const [pendingAtlassian, setPendingAtlassian] = useState<AtlassianConnection | null>(null);
  // Pending Git connection (set after form is filled, before user confirms)
  const [pendingGit, setPendingGit] = useState<GitConnection | null>(null);
  // Ref for GitPATStep's submit function so the wizard's Next button calls it
  const gitSubmitRef = useRef<(() => void) | null>(null);
  const { saveAtlassian, saveGitConnection, atlassian } = useConnections();

  function reset() {
    setActiveStep(0);
    setConnector(null);
    setPendingAtlassian(null);
    setPendingGit(null);
    gitSubmitRef.current = null;
  }

  function handleClose() {
    reset();
    onClose();
  }

  // -------------------------------------------------------------------------
  // Step navigation
  // -------------------------------------------------------------------------

  function handleNext() {
    if (activeStep === 0) {
      // Must pick a connector to advance
      if (!connector) return;
      setActiveStep(1);
      return;
    }

    if (activeStep === 1) {
      if (connector === 'git') {
        // Delegate to the form's submit; onReady fires when valid
        if (gitSubmitRef.current) gitSubmitRef.current();
        return;
      }
      // Atlassian: Next is disabled until OAuth completes (pendingAtlassian is set)
      if (connector === 'atlassian' && pendingAtlassian) {
        setActiveStep(2);
      }
      return;
    }

    if (activeStep === 2) {
      // Final save
      if (connector === 'atlassian' && pendingAtlassian) {
        saveAtlassian(pendingAtlassian);
      } else if (connector === 'git' && pendingGit) {
        saveGitConnection(pendingGit);
      }
      handleClose();
    }
  }

  function handleBack() {
    if (activeStep > 0) setActiveStep((s) => s - 1);
  }

  // Called by AtlassianStep after re-reading from hook (the callback saved it)
  function onAtlassianConnectedFromStorage(c: AtlassianConnection) {
    setPendingAtlassian(c);
    setActiveStep(2);
  }

  // Called by GitPATStep when form validates
  function onGitReady(c: GitConnection) {
    setPendingGit(c);
    setActiveStep(2);
  }

  // -------------------------------------------------------------------------
  // Atlassian step special: read from hook after OAuth
  // The OAuthCallback already persisted the connection, so we pull from state.
  // -------------------------------------------------------------------------
  function handleAtlassianConnected() {
    // atlassian is the live hook value — after postMessage the storage event
    // fires and the hook re-reads localStorage.
    // We pass the function reference; the AtlassianStep calls it.
    if (atlassian) {
      onAtlassianConnectedFromStorage(atlassian);
    } else {
      // Optimistically advance; saveAtlassian was already called by the callback.
      setActiveStep(2);
    }
  }

  // -------------------------------------------------------------------------
  // Next button disabled logic
  // -------------------------------------------------------------------------

  function isNextDisabled(): boolean {
    if (activeStep === 0) return connector === null;
    if (activeStep === 1 && connector === 'atlassian') {
      // Atlassian: Next is handled by the step itself (postMessage)
      return true;
    }
    return false;
  }

  // -------------------------------------------------------------------------
  // Step 3 summary
  // -------------------------------------------------------------------------

  function renderConfirm() {
    if (connector === 'atlassian') {
      const c = pendingAtlassian ?? atlassian;
      if (!c)
        return (
          <Typography color="text.secondary">
            Atlassian connection will be saved.
          </Typography>
        );
      return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <Typography variant="subtitle2">Atlassian connection</Typography>
          <Typography variant="body2">
            <strong>Site:</strong> {c.site_name}
          </Typography>
          <Typography variant="body2">
            <strong>Cloud ID:</strong> {c.cloud_id}
          </Typography>
          <Typography variant="body2">
            <strong>Sites available:</strong> {c.accessible_resources.length}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Tokens are stored in your browser's localStorage. They are never sent to
            the Wikis backend.
          </Typography>
        </Box>
      );
    }

    if (connector === 'git' && pendingGit) {
      return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <Typography variant="subtitle2">Git PAT connection</Typography>
          <Typography variant="body2">
            <strong>Repository:</strong> {pendingGit.repo_url}
          </Typography>
          <Typography variant="body2">
            <strong>Branch:</strong> {pendingGit.branch}
          </Typography>
          {pendingGit.label && (
            <Typography variant="body2">
              <strong>Label:</strong> {pendingGit.label}
            </Typography>
          )}
          <Typography variant="body2">
            <strong>PAT:</strong> {'•'.repeat(12)}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            The PAT is stored in your browser's localStorage. It is never sent to the
            Wikis backend directly — only passed as a credential during wiki generation.
          </Typography>
        </Box>
      );
    }

    return null;
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ pb: 0 }}>Add Connection</DialogTitle>

      <DialogContent sx={{ pt: 2 }}>
        <Stepper activeStep={activeStep} sx={{ mb: 3 }}>
          {STEPS.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {/* Step 1: Pick connector */}
        {activeStep === 0 && (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <ConnectorTile
              selected={connector === 'atlassian'}
              onClick={() => setConnector('atlassian')}
              logo={
                <Box
                  sx={{
                    width: 40,
                    height: 40,
                    borderRadius: 1,
                    bgcolor: '#0052CC',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#fff',
                    fontWeight: 700,
                    fontSize: '0.75rem',
                    flexShrink: 0,
                  }}
                >
                  AT
                </Box>
              }
              label="Atlassian (Confluence + Jira)"
              description="OAuth 2.0 with PKCE — no password stored, tokens refresh automatically."
            />

            <ConnectorTile
              selected={connector === 'git'}
              onClick={() => setConnector('git')}
              logo={
                <Box
                  sx={{
                    width: 40,
                    height: 40,
                    borderRadius: 1,
                    bgcolor: 'action.selected',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 700,
                    fontSize: '0.75rem',
                    color: 'text.secondary',
                    flexShrink: 0,
                  }}
                >
                  GIT
                </Box>
              }
              label="Git repository (PAT)"
              description="GitHub, GitLab, Bitbucket, or any self-hosted Git server."
            />
          </Box>
        )}

        {/* Step 2: Configure */}
        {activeStep === 1 && connector === 'atlassian' && (
          <AtlassianStep onConnected={handleAtlassianConnected} />
        )}

        {activeStep === 1 && connector === 'git' && (
          <GitPATStep onReady={onGitReady} submitRef={gitSubmitRef} />
        )}

        {/* Step 3: Confirm */}
        {activeStep === 2 && renderConfirm()}
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 2 }}>
        {activeStep > 0 && (
          <Button onClick={handleBack} disabled={activeStep === 1 && connector === 'atlassian'}>
            Back
          </Button>
        )}
        <Box sx={{ flex: 1 }} />
        <Button onClick={handleClose}>Cancel</Button>
        {/* Hide Next on Atlassian step 2 — flow is driven by the OAuth popup */}
        {!(activeStep === 1 && connector === 'atlassian') && (
          <Button
            variant="contained"
            onClick={handleNext}
            disabled={isNextDisabled()}
          >
            {activeStep === STEPS.length - 1 ? 'Save' : 'Next'}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}
