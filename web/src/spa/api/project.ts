import { apiRequest } from './client';
import type { components } from './types.generated';
import { subscribeAskSSE, subscribeResearchSSE } from './sse';
import type { AskSSEEvent, ResearchSSEEvent } from './sse';

type WikiSummary = components['schemas']['WikiSummary'];
type ChatMessage = components['schemas']['ChatMessage'];

export interface ProjectResponse {
  id: string;
  name: string;
  description: string | null;
  visibility: 'personal' | 'shared';
  owner_id: string;
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

/**
 * Stream ask (fast Q&A) events scoped to a project.
 * Passes project_id instead of wiki_id so the backend queries all member wikis.
 */
export const askProject = (
  projectId: string,
  question: string,
  onEvent: (event: AskSSEEvent) => void,
  onDone: () => void,
  onError: (e: unknown) => void,
  chatHistory?: ChatMessage[],
): (() => void) => {
  return subscribeAskSSE(
    '/api/v1/ask',
    { project_id: projectId, question, chat_history: chatHistory ?? [], stream: true },
    onEvent,
    onDone,
    onError,
  );
};

/**
 * Stream deep-research events scoped to a project.
 */
export const researchProject = (
  projectId: string,
  question: string,
  onEvent: (event: ResearchSSEEvent) => void,
  onDone: () => void,
  onError: (e: unknown) => void,
): (() => void) => {
  return subscribeResearchSSE(
    '/api/v1/research',
    { project_id: projectId, question, stream: true },
    onEvent,
    onDone,
    onError,
  );
};

/**
 * Stream code-map events for a project.
 */
export const mapProject = (
  projectId: string,
  entryPoints: string[],
  onEvent: (event: ResearchSSEEvent) => void,
  onDone: () => void,
  onError: (e: unknown) => void,
): (() => void) => {
  return subscribeResearchSSE(
    `/api/v1/projects/${encodeURIComponent(projectId)}/map`,
    { entry_points: entryPoints },
    onEvent,
    onDone,
    onError,
  );
};

export type { AskSSEEvent, ResearchSSEEvent };
