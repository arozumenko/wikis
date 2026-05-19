/**
 * Shared state shape for the AddSource wizard (#208).
 *
 * One canonical form-state object lives in the wizard container; each step
 * reads/writes the slice it owns. Prevents prop-drilling 15 setters from
 * the original GenerateForm and makes it trivial to remount the wizard with
 * a partial preset (e.g. URL paste from the dashboard search box).
 */

import type { WikiSourceType } from '../../api/wiki';

export type PatSource = 'none' | 'paste';

export interface GitFormState {
  repo_url: string;
  branch: string;
  patSource: PatSource;
  pastedPat: string;
}

export interface ConfluenceFormState {
  space_keys: string[];
}

export interface JiraFormState {
  jql: string;
}

/**
 * Atlassian credentials chosen at the Configure step.  The wizard supports
 * two authentication shapes:
 *
 * - ``"oauth"``: uses the in-app OAuth flow; credentials come from
 *   ``useConnections().atlassian`` and are NOT mirrored into this form
 *   state (kept as the single source of truth in the connections store).
 * - ``"api_token"``: HTTP Basic auth using an Atlassian-issued API token
 *   (see https://id.atlassian.com/manage-profile/security/api-tokens).
 *   ``siteUrl``, ``email``, and ``apiToken`` are required.
 */
export type AtlassianAuthMode = 'oauth' | 'api_token';

export interface AtlassianBasicAuthFormState {
  siteUrl: string;
  email: string;
  apiToken: string;
}

export interface WizardFormData {
  source_type: WikiSourceType;
  git: GitFormState;
  confluence: ConfluenceFormState;
  jira: JiraFormState;
  atlassianAuthMode: AtlassianAuthMode;
  atlassianBasic: AtlassianBasicAuthFormState;
  wiki_title: string;
}

export const INITIAL_FORM_DATA: WizardFormData = {
  source_type: 'git',
  git: {
    repo_url: '',
    branch: 'main',
    patSource: 'none',
    pastedPat: '',
  },
  confluence: { space_keys: [] },
  jira: { jql: 'ORDER BY created DESC' },
  atlassianAuthMode: 'oauth',
  atlassianBasic: { siteUrl: '', email: '', apiToken: '' },
  wiki_title: '',
};

/**
 * Step index enum mirrored as integers for MUI Stepper.
 *
 * Kept centralised so step components and the container agree without
 * magic numbers floating around.
 */
export const WIZARD_STEPS = ['Connector', 'Configure', 'Scan', 'Confirm'] as const;
export type WizardStepIndex = 0 | 1 | 2 | 3;
