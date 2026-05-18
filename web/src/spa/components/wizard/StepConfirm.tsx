/**
 * Step 4 of the AddSource wizard (#208): summary + final submit.
 *
 * Echoes every user choice (connector, scope, auth mode, planner) plus
 * whatever the Scan step returned, then exposes a single "Add source"
 * button. The button is intentionally the only submit path in the wizard
 * — back navigation never submits, advance buttons only validate +
 * advance. Keeps the dispatch site obvious and prevents a "did the wizard
 * already start generation?" footgun.
 */

import {
  Alert,
  Box,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import type { ScanResponse } from '../../api/wiki';
import type { WizardFormData } from './types';

interface StepConfirmProps {
  data: WizardFormData;
  onChange: (next: WizardFormData) => void;
  scanResult: ScanResponse | null;
  scanSkipped: boolean;
  submitError: string | null;
  disabled?: boolean;
}

export function StepConfirm({
  data,
  onChange,
  scanResult,
  scanSkipped,
  submitError,
  disabled,
}: StepConfirmProps) {
  const summaryRows = buildSummaryRows(data);

  return (
    <Box sx={{ mt: 0.5 }} data-testid="step-confirm">
      <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
        Source summary
      </Typography>
      <Stack divider={<Divider flexItem />} spacing={1}>
        {summaryRows.map(([label, value]) => (
          <Stack direction="row" key={label} spacing={2} sx={{ py: 0.5 }}>
            <Typography variant="body2" color="text.secondary" sx={{ minWidth: 140 }}>
              {label}
            </Typography>
            <Typography variant="body2" sx={{ wordBreak: 'break-all' }}>
              {value}
            </Typography>
          </Stack>
        ))}
      </Stack>

      {scanSkipped && (
        <Alert severity="info" sx={{ mt: 2 }} data-testid="confirm-scan-skipped">
          Submitting without a preview — the source was not validated upfront.
        </Alert>
      )}
      {!scanSkipped && scanResult?.reachable === false && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          The scan reported this source as unreachable. Submission will likely fail.
        </Alert>
      )}
      {!scanSkipped && scanResult?.preview && (
        <Alert severity="success" sx={{ mt: 2 }} data-testid="confirm-scan-stats">
          Preview: {scanResult.preview.file_count.toLocaleString()} files,{' '}
          {scanResult.preview.top_paths.length} top-level entries.
        </Alert>
      )}

      <Divider sx={{ my: 2 }} />

      <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
        Wiki options
      </Typography>

      <TextField
        label="Wiki title (optional)"
        value={data.wiki_title}
        onChange={(e) => onChange({ ...data, wiki_title: e.target.value })}
        fullWidth
        margin="dense"
        disabled={disabled}
        helperText="Leave blank to derive from the source"
        inputProps={{ 'data-testid': 'wiki-title' }}
      />

      <FormControl fullWidth margin="dense" disabled={disabled}>
        <InputLabel id="planner-mode-label">Structure planner</InputLabel>
        <Select
          labelId="planner-mode-label"
          value={data.plannerMode}
          label="Structure planner"
          onChange={(e) =>
            onChange({ ...data, plannerMode: e.target.value as WizardFormData['plannerMode'] })
          }
          inputProps={{ 'data-testid': 'planner-mode-select' }}
        >
          <MenuItem value="agentic">Agentic (LLM-driven outline)</MenuItem>
          <MenuItem value="graph_clustering">Graph clustering (Leiden)</MenuItem>
        </Select>
      </FormControl>

      {submitError && (
        <Alert severity="error" sx={{ mt: 2 }} data-testid="confirm-submit-error">
          {submitError}
        </Alert>
      )}
    </Box>
  );
}

function buildSummaryRows(data: WizardFormData): [string, string][] {
  const rows: [string, string][] = [['Connector', data.source_type]];
  if (data.source_type === 'git') {
    rows.push(['Repository', data.git.repo_url]);
    rows.push(['Branch', data.git.branch || 'main']);
    const authLabel =
      data.git.patSource === 'none'
        ? 'No auth (public)'
        : data.git.patSource === 'stored'
          ? 'Stored PAT'
          : 'Pasted PAT (not stored)';
    rows.push(['Auth', authLabel]);
  } else if (data.source_type === 'confluence') {
    rows.push(['Spaces', data.confluence.space_keys.join(', ') || '(none)']);
  } else if (data.source_type === 'jira') {
    rows.push(['JQL', data.jira.jql]);
  }
  return rows;
}
