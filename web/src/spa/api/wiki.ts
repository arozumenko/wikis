import { apiRequest } from './client';
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
  apiRequest<{ wiki_id: string; visibility: string }>(`/api/v1/wikis/${encodeURIComponent(wikiId)}/visibility`, {
    method: 'PATCH',
    body: JSON.stringify({ visibility }),
  });

export const healthCheck = () => apiRequest<HealthResponse>('/api/v1/health');
