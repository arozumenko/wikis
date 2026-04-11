/**
 * Tests for search API client functions.
 * Uses Jest with manual mocks for the fetch-based apiRequest helper.
 */

// Mock the client module so we can inspect calls without real HTTP
jest.mock('../client', () => ({
  apiRequest: jest.fn(),
  getAuthToken: jest.fn().mockResolvedValue('mock-token'),
  ApiError: class ApiError extends Error {
    status: number;
    body: unknown;
    constructor(status: number, body: unknown) {
      super(`API error ${status}`);
      this.name = 'ApiError';
      this.status = status;
      this.body = body;
    }
  },
}));

import { apiRequest } from '../client';
import {
  searchWiki,
  listWikiPages,
  getWikiPage,
  getPageNeighbors,
  searchProject,
} from '../search';

const mockApiRequest = apiRequest as jest.MockedFunction<typeof apiRequest>;

beforeEach(() => {
  mockApiRequest.mockReset();
});

// ---------------------------------------------------------------------------
// searchWiki
// ---------------------------------------------------------------------------

describe('searchWiki', () => {
  it('calls the correct URL with all provided params', async () => {
    mockApiRequest.mockResolvedValueOnce({ query: 'auth', results: [], wiki_summary: [] });

    await searchWiki('wiki-123', 'auth', 2, 5);

    expect(mockApiRequest).toHaveBeenCalledTimes(1);
    const url: string = mockApiRequest.mock.calls[0][0] as string;
    expect(url).toContain('/api/v1/wikis/wiki-123/search');
    expect(url).toContain('q=auth');
    expect(url).toContain('hop_depth=2');
    expect(url).toContain('top_k=5');
  });

  it('defaults hop_depth to 1 and top_k to 10 when omitted', async () => {
    mockApiRequest.mockResolvedValueOnce({ query: 'auth', results: [], wiki_summary: [] });

    await searchWiki('wiki-123', 'auth');

    const url: string = mockApiRequest.mock.calls[0][0] as string;
    expect(url).toContain('hop_depth=1');
    expect(url).toContain('top_k=10');
  });

  it('URL-encodes wikiId in the path', async () => {
    mockApiRequest.mockResolvedValueOnce({ query: 'q', results: [], wiki_summary: [] });

    await searchWiki('wiki/special id', 'q');

    const url: string = mockApiRequest.mock.calls[0][0] as string;
    expect(url).toContain(encodeURIComponent('wiki/special id'));
  });
});

// ---------------------------------------------------------------------------
// listWikiPages
// ---------------------------------------------------------------------------

describe('listWikiPages', () => {
  it('calls the correct URL for the given wikiId', async () => {
    mockApiRequest.mockResolvedValueOnce({ wiki_id: 'wiki-abc', pages: [] });

    await listWikiPages('wiki-abc');

    expect(mockApiRequest).toHaveBeenCalledWith('/api/v1/wikis/wiki-abc/pages');
  });
});

// ---------------------------------------------------------------------------
// getWikiPage
// ---------------------------------------------------------------------------

describe('getWikiPage', () => {
  it('URL-encodes page title with spaces and special chars', async () => {
    mockApiRequest.mockResolvedValueOnce({
      wiki_id: 'w1',
      page_title: 'Auth & Security',
      content: '',
      sections: [],
    });

    await getWikiPage('w1', 'Auth & Security');

    const url: string = mockApiRequest.mock.calls[0][0] as string;
    expect(url).toBe(`/api/v1/wikis/w1/pages/${encodeURIComponent('Auth & Security')}`);
  });

  it('calls the correct URL for a plain page title', async () => {
    mockApiRequest.mockResolvedValueOnce({
      wiki_id: 'w2',
      page_title: 'Overview',
      content: 'text',
      sections: ['Intro'],
    });

    await getWikiPage('w2', 'Overview');

    expect(mockApiRequest).toHaveBeenCalledWith('/api/v1/wikis/w2/pages/Overview');
  });
});

// ---------------------------------------------------------------------------
// getPageNeighbors
// ---------------------------------------------------------------------------

describe('getPageNeighbors', () => {
  it('includes hop_depth query param in the URL', async () => {
    mockApiRequest.mockResolvedValueOnce({
      wiki_id: 'w1',
      page_title: 'Overview',
      links_to: [],
      linked_from: [],
    });

    await getPageNeighbors('w1', 'Overview', 3);

    const url: string = mockApiRequest.mock.calls[0][0] as string;
    expect(url).toContain('hop_depth=3');
    expect(url).toContain(`/api/v1/wikis/w1/pages/${encodeURIComponent('Overview')}/neighbors`);
  });

  it('defaults hop_depth to 1 when not provided', async () => {
    mockApiRequest.mockResolvedValueOnce({
      wiki_id: 'w1',
      page_title: 'Overview',
      links_to: [],
      linked_from: [],
    });

    await getPageNeighbors('w1', 'Overview');

    const url: string = mockApiRequest.mock.calls[0][0] as string;
    expect(url).toContain('hop_depth=1');
  });
});

// ---------------------------------------------------------------------------
// searchProject
// ---------------------------------------------------------------------------

describe('searchProject', () => {
  it('calls the correct URL with all provided params', async () => {
    mockApiRequest.mockResolvedValueOnce({ query: 'db', results: [], wiki_summary: [] });

    await searchProject('proj-99', 'db', 2, 20);

    const url: string = mockApiRequest.mock.calls[0][0] as string;
    expect(url).toContain('/api/v1/projects/proj-99/search');
    expect(url).toContain('q=db');
    expect(url).toContain('hop_depth=2');
    expect(url).toContain('top_k=20');
  });

  it('defaults hop_depth to 1 and top_k to 10 when omitted', async () => {
    mockApiRequest.mockResolvedValueOnce({ query: 'db', results: [], wiki_summary: [] });

    await searchProject('proj-99', 'db');

    const url: string = mockApiRequest.mock.calls[0][0] as string;
    expect(url).toContain('hop_depth=1');
    expect(url).toContain('top_k=10');
  });
});

// ---------------------------------------------------------------------------
// Error propagation
// ---------------------------------------------------------------------------

describe('error handling', () => {
  it('propagates ApiError thrown by apiRequest', async () => {
    const { ApiError } = jest.requireMock('../client') as {
      ApiError: new (status: number, body: unknown) => Error & { status: number };
    };
    mockApiRequest.mockRejectedValueOnce(new ApiError(404, { detail: 'not found' }));

    await expect(searchWiki('missing', 'q')).rejects.toMatchObject({
      status: 404,
      body: { detail: 'not found' },
    });
  });
});

// ---------------------------------------------------------------------------
// Auth headers — apiRequest is the single auth-aware layer
// ---------------------------------------------------------------------------

describe('auth header delegation', () => {
  it('all functions delegate to apiRequest, which adds auth headers', async () => {
    // Every function in search.ts goes through apiRequest — verify it is called
    // for each function so that auth is guaranteed by the shared client.
    mockApiRequest.mockResolvedValue({ query: '', results: [], wiki_summary: [] });

    await searchWiki('w', 'q');
    await searchProject('p', 'q');

    mockApiRequest.mockResolvedValue({ wiki_id: 'w', pages: [] });
    await listWikiPages('w');

    mockApiRequest.mockResolvedValue({
      wiki_id: 'w',
      page_title: 't',
      content: '',
      sections: [],
    });
    await getWikiPage('w', 't');

    mockApiRequest.mockResolvedValue({
      wiki_id: 'w',
      page_title: 't',
      links_to: [],
      linked_from: [],
    });
    await getPageNeighbors('w', 't');

    // 5 calls total — one per function
    expect(mockApiRequest).toHaveBeenCalledTimes(5);
  });
});
