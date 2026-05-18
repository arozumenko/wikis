/**
 * AddSourceWizard — 4-step source-ingestion flow (#208).
 *
 * Replaces the dashboard's tabbed ``GenerateForm`` dialog. Four steps:
 *   1. ConnectorPicker  — select source type.
 *   2. Configure        — per-connector form fields.
 *   3. Scan             — preview via POST /api/v1/sources/scan (skippable).
 *   4. Confirm          — summary + final submit via POST /api/v1/wikis.
 *
 * Design notes
 * ------------
 * Single canonical ``formData`` state lives here; child steps receive
 * only the slice they own plus an ``onChange`` that merges back. Per-step
 * validation is computed in this container (the connector forms remain
 * dumb), so the Next button is gated centrally and the same validator
 * decides whether the Scan step can build a ``ScanRequest``.
 *
 * Cancellation: an in-flight scan tied to Step 3 is not explicitly aborted
 * when the user navigates away — the response either lands and updates
 * stale state harmlessly, or is discarded by the next mount of StepScan
 * (which re-runs on mount).
 */

import { useCallback, useMemo, useState } from 'react';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Stack,
  Step,
  StepLabel,
  Stepper,
} from '@mui/material';
import {
  generateWikiMultiSource,
  type AtlassianAuth,
  type GenerateWikiMultiSourceRequest,
  type GenerateWikiResponse,
  type ScanRequest,
  type ScanResponse,
} from '../../api/wiki';
import { useConnections } from '../../hooks/useConnections';
import { StepConnector } from './StepConnector';
import { StepConfigure } from './StepConfigure';
import { StepScan } from './StepScan';
import { StepConfirm } from './StepConfirm';
import {
  INITIAL_FORM_DATA,
  WIZARD_STEPS,
  type WizardFormData,
  type WizardStepIndex,
} from './types';

interface AddSourceWizardProps {
  open: boolean;
  onClose: () => void;
  initialUrl?: string;
  /** Called after a successful generation has been initiated. */
  onSuccess: (response: GenerateWikiResponse) => void;
  /**
   * Called when the backend returns 409 (a wiki for this source already
   * exists). The handler is expected to close the wizard and navigate to
   * the existing wiki. Optional — if omitted, the wizard surfaces the
   * conflict as an inline error message and stays open.
   */
  onAlreadyExists?: (existingWikiId: string | null) => void;
}

const URL_PATTERN = /^(https?:\/\/|git@|file:\/\/|\/)/;

function isUrlish(value: string): boolean {
  return URL_PATTERN.test(value.trim());
}

export function AddSourceWizard({
  open,
  onClose,
  initialUrl,
  onSuccess,
  onAlreadyExists,
}: AddSourceWizardProps) {
  const [step, setStep] = useState<WizardStepIndex>(0);
  const [formData, setFormData] = useState<WizardFormData>(() => ({
    ...INITIAL_FORM_DATA,
    git: { ...INITIAL_FORM_DATA.git, repo_url: initialUrl ?? '' },
  }));
  const [scanResult, setScanResult] = useState<ScanResponse | null>(null);
  const [scanSkipped, setScanSkipped] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const { atlassian, connections, refreshAtlassianIfNeeded } = useConnections();
  const gitConnections = useMemo(
    () => connections.filter((c) => c.provider === 'git'),
    [connections],
  );

  // ---------------------------------------------------------------------
  // Per-step validation
  // ---------------------------------------------------------------------

  const urlError = useMemo<string | null>(() => {
    if (formData.source_type !== 'git') return null;
    const url = formData.git.repo_url.trim();
    if (!url) return 'Repository URL is required';
    if (!isUrlish(url))
      return 'Must be a valid URL (https://, git@, file://, or local /path)';
    return null;
  }, [formData.source_type, formData.git.repo_url]);

  const spaceKeysError = useMemo<string | null>(() => {
    if (formData.source_type !== 'confluence') return null;
    if (formData.confluence.space_keys.length === 0) return 'At least one space key is required';
    return null;
  }, [formData.source_type, formData.confluence.space_keys]);

  const jqlError = useMemo<string | null>(() => {
    if (formData.source_type !== 'jira') return null;
    if (!formData.jira.jql.trim()) return 'JQL query is required';
    return null;
  }, [formData.source_type, formData.jira.jql]);

  const atlassianRequired =
    formData.source_type === 'confluence' || formData.source_type === 'jira';
  const atlassianMissing = atlassianRequired && !atlassian;

  const configureValid = useMemo(() => {
    if (atlassianMissing) return false;
    if (formData.source_type === 'git') {
      if (urlError) return false;
      if (formData.git.patSource === 'stored') {
        if (gitConnections.length === 0) return false;
        if (!formData.git.selectedPatId) return false;
      }
      return true;
    }
    if (formData.source_type === 'confluence') return !spaceKeysError;
    if (formData.source_type === 'jira') return !jqlError;
    return false;
  }, [
    atlassianMissing,
    formData.source_type,
    formData.git.patSource,
    formData.git.selectedPatId,
    gitConnections.length,
    urlError,
    spaceKeysError,
    jqlError,
  ]);

  // ---------------------------------------------------------------------
  // Build a ScanRequest from current form data. The Scan step calls this
  // (via the ``buildScanRequest`` prop) at mount; returning null tells the
  // step to render an "invalid configuration" message rather than POST.
  //
  // Atlassian token refresh is intentionally NOT done here — Step 3 only
  // hits the (still-501) Atlassian backend in #211, and a pre-emptive
  // refresh now would spend tokens unnecessarily. The submit path below
  // does the refresh once, immediately before POST /wikis.
  // ---------------------------------------------------------------------

  const buildScanRequest = useCallback((): ScanRequest | null => {
    if (!configureValid) return null;
    if (formData.source_type === 'git') {
      let pat: string | null = null;
      if (formData.git.patSource === 'stored' && formData.git.selectedPatId) {
        pat = gitConnections.find((c) => c.id === formData.git.selectedPatId)?.pat ?? null;
      } else if (formData.git.patSource === 'paste') {
        pat = formData.git.pastedPat || null;
      }
      return {
        source_type: 'git',
        scope: { repo_url: formData.git.repo_url.trim(), branch: formData.git.branch || 'main' },
        auth: { pat },
      };
    }
    if (!atlassian) return null;
    const baseUrl = atlassian.accessible_resources[0]?.url ?? atlassian.site_name;
    const atlassianAuth: AtlassianAuth = {
      access_token: atlassian.access_token,
      refresh_token: atlassian.refresh_token,
      client_id: null,
    };
    if (formData.source_type === 'confluence') {
      return {
        source_type: 'confluence',
        scope: { base_url: baseUrl, space_keys: formData.confluence.space_keys },
        auth: atlassianAuth,
      };
    }
    return {
      source_type: 'jira',
      scope: { base_url: baseUrl, jql: formData.jira.jql },
      auth: atlassianAuth,
    };
  }, [
    configureValid,
    formData.source_type,
    formData.git,
    formData.confluence.space_keys,
    formData.jira.jql,
    gitConnections,
    atlassian,
  ]);

  // ---------------------------------------------------------------------
  // Step navigation
  // ---------------------------------------------------------------------

  const goNext = () => setStep((s) => Math.min(3, s + 1) as WizardStepIndex);
  const goBack = () => setStep((s) => Math.max(0, s - 1) as WizardStepIndex);

  const skipScan = () => {
    setScanSkipped(true);
    setScanResult(null);
    setStep(3);
  };

  // ---------------------------------------------------------------------
  // Final submit
  // ---------------------------------------------------------------------

  const handleSubmit = useCallback(async () => {
    setSubmitError(null);
    setSubmitting(true);
    try {
      let body: GenerateWikiMultiSourceRequest;
      if (formData.source_type === 'git') {
        let pat: string | null = null;
        if (formData.git.patSource === 'stored' && formData.git.selectedPatId) {
          pat = gitConnections.find((c) => c.id === formData.git.selectedPatId)?.pat ?? null;
        } else if (formData.git.patSource === 'paste') {
          pat = formData.git.pastedPat || null;
        }
        body = {
          source_type: 'git',
          scope: {
            repo_url: formData.git.repo_url.trim(),
            branch: formData.git.branch || 'main',
          },
          auth: { pat },
          ...(formData.wiki_title ? { wiki_title: formData.wiki_title } : {}),
          structure_planner: formData.plannerMode,
        };
      } else {
        const fresh = await refreshAtlassianIfNeeded();
        if (!fresh) {
          setSubmitError('Atlassian connection lost. Please reconnect in Settings.');
          setSubmitting(false);
          return;
        }
        const baseUrl = fresh.accessible_resources[0]?.url ?? fresh.site_name;
        const auth: AtlassianAuth = {
          access_token: fresh.access_token,
          refresh_token: fresh.refresh_token,
          client_id: null,
        };
        if (formData.source_type === 'confluence') {
          body = {
            source_type: 'confluence',
            scope: { base_url: baseUrl, space_keys: formData.confluence.space_keys },
            auth,
            ...(formData.wiki_title ? { wiki_title: formData.wiki_title } : {}),
            structure_planner: formData.plannerMode,
          };
        } else {
          body = {
            source_type: 'jira',
            scope: { base_url: baseUrl, jql: formData.jira.jql },
            auth,
            ...(formData.wiki_title ? { wiki_title: formData.wiki_title } : {}),
            structure_planner: formData.plannerMode,
          };
        }
      }
      const response = await generateWikiMultiSource(body);
      onSuccess(response);
    } catch (err: unknown) {
      // 409 — wiki already exists. The backend includes the existing
      // ``wiki_id`` in the structured detail; hand it to the parent so it
      // can navigate to the existing wiki (matches the legacy flow). If
      // no handler was wired, fall back to an inline error.
      if (err && typeof err === 'object' && 'status' in err) {
        const status = (err as { status: number }).status;
        if (status === 409) {
          const body = (err as { body?: { detail?: { wiki_id?: string } } }).body;
          const existingId = body?.detail?.wiki_id ?? null;
          if (onAlreadyExists) {
            onAlreadyExists(existingId);
          } else {
            setSubmitError('A wiki for this source already exists.');
          }
          return;
        }
      }
      setSubmitError(
        err instanceof Error ? err.message : 'Failed to start generation. Please try again.',
      );
    } finally {
      setSubmitting(false);
    }
  }, [formData, gitConnections, refreshAtlassianIfNeeded, onSuccess, onAlreadyExists]);

  // ---------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------

  const canAdvance =
    (step === 0) ||
    (step === 1 && configureValid) ||
    (step === 2) || // Scan step always allows advance — preview is optional
    (step === 3); // Confirm uses Submit button, not Next

  return (
    <Dialog
      open={open}
      onClose={submitting ? undefined : onClose}
      maxWidth="sm"
      fullWidth
      data-testid="add-source-wizard"
    >
      <DialogTitle>Add a source</DialogTitle>
      <DialogContent>
        <Stepper activeStep={step} sx={{ mb: 2 }}>
          {WIZARD_STEPS.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {step === 0 && (
          <StepConnector
            selected={formData.source_type}
            onSelect={(source_type) => {
              setFormData((d) => ({ ...d, source_type }));
              // Auto-advance on pick (cartograph behavior).
              setStep(1);
            }}
          />
        )}
        {step === 1 && (
          <StepConfigure
            data={formData}
            onChange={setFormData}
            urlError={urlError}
            spaceKeysError={spaceKeysError}
            jqlError={jqlError}
            disabled={submitting}
          />
        )}
        {step === 2 && (
          <StepScan
            buildScanRequest={buildScanRequest}
            onScanComplete={(result) => {
              setScanResult(result);
              setScanSkipped(false);
            }}
          />
        )}
        {step === 3 && (
          <StepConfirm
            data={formData}
            onChange={setFormData}
            scanResult={scanResult}
            scanSkipped={scanSkipped}
            submitError={submitError}
            disabled={submitting}
          />
        )}
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Stack direction="row" spacing={1} sx={{ flex: 1 }}>
          {step > 0 && step < 3 && (
            <Button onClick={goBack} disabled={submitting} data-testid="wizard-back">
              Back
            </Button>
          )}
          {step === 2 && (
            <Button onClick={skipScan} disabled={submitting} data-testid="wizard-skip-scan">
              Skip preview
            </Button>
          )}
        </Stack>
        <Button onClick={onClose} disabled={submitting}>
          Cancel
        </Button>
        {step < 3 && step > 0 && (
          <Button
            variant="contained"
            onClick={goNext}
            disabled={!canAdvance || submitting}
            data-testid="wizard-next"
          >
            Next
          </Button>
        )}
        {step === 3 && (
          <Button
            variant="contained"
            onClick={() => void handleSubmit()}
            disabled={submitting}
            data-testid="wizard-submit"
          >
            {submitting ? 'Starting…' : 'Add source'}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}
