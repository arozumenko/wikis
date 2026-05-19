/**
 * Step 2 of the AddSource wizard (#208): connector-specific configuration.
 *
 * Pure dispatch component — delegates to the appropriate ``*Configure``
 * form based on ``data.source_type``. Validation (URL shape, JQL non-empty,
 * space keys non-empty) is computed by the container and surfaced via the
 * per-connector error props so the Next button can be gated centrally.
 */

import { Box } from '@mui/material';
import { GitConfigure } from './connectors/GitConfigure';
import { ConfluenceConfigure } from './connectors/ConfluenceConfigure';
import { JiraConfigure } from './connectors/JiraConfigure';
import type { WizardFormData } from './types';

interface StepConfigureProps {
  data: WizardFormData;
  onChange: (next: WizardFormData) => void;
  urlError: string | null;
  spaceKeysError: string | null;
  jqlError: string | null;
  disabled?: boolean;
}

export function StepConfigure({
  data,
  onChange,
  urlError,
  spaceKeysError,
  jqlError,
  disabled,
}: StepConfigureProps) {
  return (
    <Box sx={{ mt: 0.5 }}>
      {data.source_type === 'git' && (
        <GitConfigure
          data={data.git}
          onChange={(git) => onChange({ ...data, git })}
          urlError={urlError}
          disabled={disabled}
        />
      )}
      {data.source_type === 'confluence' && (
        <ConfluenceConfigure
          data={data.confluence}
          onChange={(confluence) => onChange({ ...data, confluence })}
          spaceKeysError={spaceKeysError}
          disabled={disabled}
          authMode={data.atlassianAuthMode}
          onAuthModeChange={(atlassianAuthMode) => onChange({ ...data, atlassianAuthMode })}
          basicAuth={data.atlassianBasic}
          onBasicAuthChange={(atlassianBasic) => onChange({ ...data, atlassianBasic })}
        />
      )}
      {data.source_type === 'jira' && (
        <JiraConfigure
          data={data.jira}
          onChange={(jira) => onChange({ ...data, jira })}
          jqlError={jqlError}
          disabled={disabled}
          authMode={data.atlassianAuthMode}
          onAuthModeChange={(atlassianAuthMode) => onChange({ ...data, atlassianAuthMode })}
          basicAuth={data.atlassianBasic}
          onBasicAuthChange={(atlassianBasic) => onChange({ ...data, atlassianBasic })}
        />
      )}
    </Box>
  );
}
