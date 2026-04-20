import { apiRequest } from './client';
import type { components } from './types.generated';

type WikiSummary = components['schemas']['WikiSummary'];

export interface ProjectResponse {
  id: string;
  name: string;
  description: string | null;
  visibility: 'personal' | 'shared';
  owner_id: string;
  is_owner: boolean;
  created_at: string;
  wiki_count: number;
}

export interface ProjectListResponse {
  projects: ProjectResponse[];
}

export interface ProjectCreateRequest {
  name: string;
  description?: string;
  visibility: 'personal' | 'shared';
}

export const listProjects = (): Promise<ProjectListResponse> =>
  apiRequest<ProjectListResponse>('/api/v1/projects');

export const createProject = (req: ProjectCreateRequest): Promise<ProjectResponse> =>
  apiRequest<ProjectResponse>('/api/v1/projects', {
    method: 'POST',
    body: JSON.stringify(req),
  });

export const getProject = (id: string): Promise<ProjectResponse> =>
  apiRequest<ProjectResponse>(`/api/v1/projects/${encodeURIComponent(id)}`);

export const updateProject = (
  id: string,
  req: Partial<ProjectCreateRequest>,
): Promise<ProjectResponse> =>
  apiRequest<ProjectResponse>(`/api/v1/projects/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    body: JSON.stringify(req),
  });

export const deleteProject = (id: string): Promise<void> =>
  apiRequest<void>(`/api/v1/projects/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });

export const addWikiToProject = (projectId: string, wikiId: string): Promise<ProjectResponse> =>
  apiRequest<ProjectResponse>(`/api/v1/projects/${encodeURIComponent(projectId)}/wikis`, {
    method: 'POST',
    body: JSON.stringify({ wiki_id: wikiId }),
  });

export const removeWikiFromProject = (projectId: string, wikiId: string): Promise<void> =>
  apiRequest<void>(
    `/api/v1/projects/${encodeURIComponent(projectId)}/wikis/${encodeURIComponent(wikiId)}`,
    { method: 'DELETE' },
  );

export const listProjectWikis = (projectId: string): Promise<{ wikis: WikiSummary[] }> =>
  apiRequest<{ wikis: WikiSummary[] }>(
    `/api/v1/projects/${encodeURIComponent(projectId)}/wikis`,
  );

