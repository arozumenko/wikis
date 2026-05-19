import { apiRequest, clearTokenCache, getAuthToken } from './client';
import type { components } from './types.generated';

type GenerateWikiRequest = components['schemas']['GenerateWikiRequest'];
export type GenerateWikiResponse = components['schemas']['GenerateWikiResponse'];

// ---------------------------------------------------------------------------
// Multi-source wiki generation (POST /api/v1/wikis — new API shape, #117/#189)
// ---------------------------------------------------------------------------

export type WikiSourceType = 'git' | 'confluence' | 'jira';

export interface GitScope {
  repo_url: string;
  branch: string;
}

export interface ConfluenceScope {
  base_url: string;
  space_keys: string[];
}

export interface JiraScope {
  base_url: string;
  jql: string;
}

export interface GitAuth {
  pat: string | null;
}

export interface AtlassianAuth {
  access_token: string;
  refresh_token: string | null;
  client_id: string | null;
}

export interface GenerateWikiMultiSourceRequest {
  source_type: WikiSourceType;
  scope: GitScope | ConfluenceScope | JiraScope;
  auth: GitAuth | AtlassianAuth;
  wiki_title?: string;
}

export const generateWikiMultiSource = (req: GenerateWikiMultiSourceRequest) =>
  apiRequest<GenerateWikiResponse>('/api/v1/wikis', {
    method: 'POST',
    body: JSON.stringify(req),
  });

// ---------------------------------------------------------------------------
// Source scan (#207)
// ---------------------------------------------------------------------------
//
// Validates a source configuration and returns a preview without persisting
// anything server-side. Drives Step 3 of the source-ingestion wizard (#208).
// Local types declared here until the next openapi-typescript regen pulls
// them out of the backend schema.

export interface ScanRequest {
  source_type: WikiSourceType;
  scope: GitScope | ConfluenceScope | JiraScope;
  auth: GitAuth | AtlassianAuth;
}

export interface GitScanPreview {
  default_branch: string | null;
  resolved_branch: string;
  commit_hash: string | null;
  file_count: number;
  top_paths: string[];
  size_bytes: number;
}

export interface ConfluenceSpaceInfo {
  key: string;
  name: string;
  page_count: number | null;
}

export interface ConfluenceScanPreview {
  spaces: ConfluenceSpaceInfo[];
  total_pages: number | null;
}

export interface JiraScanPreview {
  matching_issues: number;
  sample_issue_keys: string[];
  jql_validated: boolean;
}

export interface ScanResponse {
  source_type: string;
  reachable: boolean;
  preview: GitScanPreview | ConfluenceScanPreview | JiraScanPreview | null;
  warnings: string[];
}

/**
 * Deterministic cache key for a scan request.
 *
 * Includes source_type, scope, and auth *presence* (not the token value)
 * so that switching auth mode (e.g. no PAT → paste a PAT) invalidates
 * the cached scan result without embedding credentials in the hash.
 *
 * Used in both AddSourceWizard (currentScopeHash memo) and StepScan
 * (hashRequest) to ensure the two sides stay in sync — a single shared
 * function eliminates the class of bug where one side forgets to include
 * auth presence while the other does (#217).
 */
export function hashScanRequest(req: ScanRequest): string {
  const authPresence =
    'pat' in req.auth ? { hasToken: req.auth.pat != null } : { hasToken: true };
  return JSON.stringify({ type: req.source_type, scope: req.scope, auth: authPresence });
}

export const scanSource = (req: ScanRequest) =>
  apiRequest<ScanResponse>('/api/v1/sources/scan', {
    method: 'POST',
    body: JSON.stringify(req),
  });
type AskRequest = components['schemas']['AskRequest'];
type AskResponse = components['schemas']['AskResponse'];
type ResearchRequest = components['schemas']['ResearchRequest'];
type ResearchResponse = components['schemas']['ResearchResponse'];
type WikiListResponse = components['schemas']['WikiListResponse'];
type DeleteWikiResponse = components['schemas']['DeleteWikiResponse'];
type HealthResponse = components['schemas']['HealthResponse'];
export type WikiSummary = components['schemas']['WikiSummary'];

export const generateWiki = (req: GenerateWikiRequest) =>
  apiRequest<GenerateWikiResponse>('/api/v1/generate', {
    method: 'POST',
    body: JSON.stringify(req),
  });

export const askQuestion = (req: AskRequest) =>
  apiRequest<AskResponse>('/api/v1/ask', {
    method: 'POST',
    body: JSON.stringify(req),
  });

export const deepResearch = (req: ResearchRequest) =>
  apiRequest<ResearchResponse>('/api/v1/research', {
    method: 'POST',
    body: JSON.stringify(req),
  });

export const listWikis = () => apiRequest<WikiListResponse>('/api/v1/wikis');

export interface WikiPage {
  id: string;
  title: string;
  order: number;
  section?: string;
  content: string;
}

export interface WikiDetail {
  wiki_id: string;
  repo_url: string;
  branch: string;
  title: string;
  page_count: number;
  created_at: string;
  indexed_at?: string | null;
  commit_hash?: string | null;
  pages: WikiPage[];
  status?: string;
  invocation_id?: string;
  requires_token?: boolean;
  description?: string | null;
  is_owner?: boolean;
}

export const getWiki = (wikiId: string) =>
  apiRequest<WikiDetail>(`/api/v1/wikis/${encodeURIComponent(wikiId)}`);

export const refreshWiki = (wikiId: string, accessToken?: string) =>
  apiRequest<GenerateWikiResponse>(`/api/v1/wikis/${encodeURIComponent(wikiId)}/refresh`, {
    method: 'POST',
    body: JSON.stringify({ access_token: accessToken ?? null }),
  });

export const deleteWiki = (id: string) =>
  apiRequest<DeleteWikiResponse>(`/api/v1/wikis/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });

export const updateWikiVisibility = (wikiId: string, visibility: 'personal' | 'shared') =>
  apiRequest<{ wiki_id: string; visibility: string }>(
    `/api/v1/wikis/${encodeURIComponent(wikiId)}/visibility`,
    {
      method: 'PATCH',
      body: JSON.stringify({ visibility }),
    },
  );

export const healthCheck = () => apiRequest<HealthResponse>('/api/v1/health');

export async function importWiki(file: File): Promise<WikiSummary> {
  const token = await getAuthToken();
  const form = new FormData();
  form.append('bundle', file);
  const res = await fetch('/api/v1/wikis/import', {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: form,
  });
  if (!res.ok) {
    if (res.status === 401) {
      clearTokenCache();
      const returnPath = window.location.pathname + window.location.search;
      window.location.href = `/login?callbackUrl=${encodeURIComponent(returnPath)}`;
      return new Promise<never>(() => {});
    }
    const text = await res.text();
    let message = `Import failed (${res.status})`;
    try {
      const json = JSON.parse(text);
      if (json.detail) message = json.detail;
    } catch (_e) {
      /* ignore JSON parse errors — fall back to HTTP status message */
    }
    throw new Error(message);
  }
  return res.json() as Promise<WikiSummary>;
}
