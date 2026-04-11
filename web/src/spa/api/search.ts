import { apiRequest } from './client';

export interface PageNeighbor {
  title: string;
  rel: 'links_to' | 'linked_from';
}

export interface SearchResultItem {
  wiki_id: string;
  wiki_name: string;
  page_title: string;
  snippet: string;
  score: number;
  neighbors: PageNeighbor[];
}

export interface WikiSummaryItem {
  wiki_id: string;
  wiki_name: string;
  match_count: number;
  relevance: number;
}

export interface WikiSearchResponse {
  query: string;
  results: SearchResultItem[];
  wiki_summary: WikiSummaryItem[];
}

export interface ProjectSearchResponse {
  query: string;
  results: SearchResultItem[];
  wiki_summary: WikiSummaryItem[];
}

export interface PageListItem {
  page_title: string;
  description: string;
  section: string | null;
}

export interface WikiPageListResponse {
  wiki_id: string;
  pages: PageListItem[];
}

export interface WikiPageResponse {
  wiki_id: string;
  page_title: string;
  content: string;
  sections: string[];
}

export interface PageNeighborsResponse {
  wiki_id: string;
  page_title: string;
  links_to: string[];
  linked_from: string[];
}

export const searchWiki = (
  wikiId: string,
  query: string,
  hopDepth = 1,
  topK = 10,
): Promise<WikiSearchResponse> => {
  const params = new URLSearchParams({
    q: query,
    hop_depth: String(hopDepth),
    top_k: String(topK),
  });
  return apiRequest<WikiSearchResponse>(
    `/api/v1/wikis/${encodeURIComponent(wikiId)}/search?${params}`,
  );
};

export const listWikiPages = (wikiId: string): Promise<WikiPageListResponse> =>
  apiRequest<WikiPageListResponse>(`/api/v1/wikis/${encodeURIComponent(wikiId)}/pages`);

export const getWikiPage = (wikiId: string, pageTitle: string): Promise<WikiPageResponse> =>
  apiRequest<WikiPageResponse>(
    `/api/v1/wikis/${encodeURIComponent(wikiId)}/pages/${encodeURIComponent(pageTitle)}`,
  );

export const getPageNeighbors = (
  wikiId: string,
  pageTitle: string,
  hopDepth = 1,
): Promise<PageNeighborsResponse> => {
  const params = new URLSearchParams({ hop_depth: String(hopDepth) });
  return apiRequest<PageNeighborsResponse>(
    `/api/v1/wikis/${encodeURIComponent(wikiId)}/pages/${encodeURIComponent(pageTitle)}/neighbors?${params}`,
  );
};

export const searchProject = (
  projectId: string,
  query: string,
  hopDepth = 1,
  topK = 10,
): Promise<ProjectSearchResponse> => {
  const params = new URLSearchParams({
    q: query,
    hop_depth: String(hopDepth),
    top_k: String(topK),
  });
  return apiRequest<ProjectSearchResponse>(
    `/api/v1/projects/${encodeURIComponent(projectId)}/search?${params}`,
  );
};
