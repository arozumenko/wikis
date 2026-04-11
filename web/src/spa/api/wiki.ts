import { apiRequest, clearTokenCache, getAuthToken } from './client';
import type { components } from './types.generated';

type GenerateWikiRequest = components['schemas']['GenerateWikiRequest'];
type GenerateWikiResponse = components['schemas']['GenerateWikiResponse'];
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

export const updateWikiDescription = (wikiId: string, description: string | null) =>
  apiRequest<WikiDetail>(`/api/v1/wikis/${encodeURIComponent(wikiId)}/description`, {
    method: 'PATCH',
    body: JSON.stringify({ description }),
  });

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
