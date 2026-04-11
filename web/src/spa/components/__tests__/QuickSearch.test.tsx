/**
 * @jest-environment jsdom
 */
import '@testing-library/jest-dom';
import React from 'react';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { QuickSearch } from '../QuickSearch';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => {
  const actual = jest.requireActual<typeof import('react-router-dom')>('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

jest.mock('../../api/search', () => ({
  searchWiki: jest.fn(),
  searchProject: jest.fn(),
}));

import { searchWiki, searchProject } from '../../api/search';

const mockSearchWiki = searchWiki as jest.MockedFunction<typeof searchWiki>;
const mockSearchProject = searchProject as jest.MockedFunction<typeof searchProject>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeResult(overrides?: Partial<import('../../api/search').SearchResultItem>) {
  return {
    wiki_id: 'wiki-1',
    wiki_name: 'Test Wiki',
    page_title: 'Overview',
    snippet: 'This is the overview page for the project.',
    score: 0.92,
    neighbors: [{ title: 'Architecture', rel: 'links_to' as const }],
    ...overrides,
  };
}

function renderComponent(props?: Partial<React.ComponentProps<typeof QuickSearch>>) {
  const defaultProps = {
    open: true,
    onClose: jest.fn(),
  };
  return render(
    <MemoryRouter>
      <QuickSearch {...defaultProps} {...props} />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  jest.useFakeTimers();
});

afterEach(() => {
  jest.runOnlyPendingTimers();
  jest.useRealTimers();
});

// ---------------------------------------------------------------------------
// 1. Renders dialog when open=true
// ---------------------------------------------------------------------------
describe('QuickSearch dialog visibility', () => {
  it('renders the dialog when open=true', () => {
    renderComponent({ open: true });
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // 2. Hidden when open=false
  // ---------------------------------------------------------------------------
  it('does not show dialog content when open=false', () => {
    renderComponent({ open: false });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// 3. searchWiki called after debounce when wikiId provided and query >= 2 chars
// ---------------------------------------------------------------------------
describe('search invocation', () => {
  it('calls searchWiki after 300ms debounce when wikiId is set and query >= 2 chars', async () => {
    mockSearchWiki.mockResolvedValueOnce({ query: 'ov', results: [], wiki_summary: [] });

    renderComponent({ wikiId: 'wiki-1' });

    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: 'ov' } });

    expect(mockSearchWiki).not.toHaveBeenCalled();

    act(() => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => {
      expect(mockSearchWiki).toHaveBeenCalledWith('wiki-1', 'ov');
    });
  });

  // ---------------------------------------------------------------------------
  // 4. searchProject called when projectId provided
  // ---------------------------------------------------------------------------
  it('calls searchProject when projectId is set', async () => {
    mockSearchProject.mockResolvedValueOnce({ query: 'db', results: [], wiki_summary: [] });

    renderComponent({ projectId: 'proj-1' });

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'db' } });

    act(() => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => {
      expect(mockSearchProject).toHaveBeenCalledWith('proj-1', 'db');
    });
  });

  // ---------------------------------------------------------------------------
  // 5. No search when query < 2 chars
  // ---------------------------------------------------------------------------
  it('does not search when query is less than 2 characters', async () => {
    renderComponent({ wikiId: 'wiki-1' });

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'a' } });

    act(() => {
      jest.advanceTimersByTime(300);
    });

    expect(mockSearchWiki).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// 6. Results rendered after search
// ---------------------------------------------------------------------------
describe('results rendering', () => {
  it('renders result page titles and snippets after a search', async () => {
    mockSearchWiki.mockResolvedValueOnce({
      query: 'over',
      results: [makeResult({ page_title: 'Overview', snippet: 'The main page.' })],
      wiki_summary: [],
    });

    renderComponent({ wikiId: 'wiki-1' });
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'over' } });

    act(() => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => {
      expect(screen.getByText('Overview')).toBeInTheDocument();
      expect(screen.getByText('The main page.')).toBeInTheDocument();
    });
  });

  it('renders neighbor chips alongside result', async () => {
    mockSearchWiki.mockResolvedValueOnce({
      query: 'over',
      results: [
        makeResult({
          neighbors: [
            { title: 'Architecture', rel: 'links_to' },
            { title: 'Setup', rel: 'linked_from' },
          ],
        }),
      ],
      wiki_summary: [],
    });

    renderComponent({ wikiId: 'wiki-1' });
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'over' } });

    act(() => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => {
      expect(screen.getByText('Architecture')).toBeInTheDocument();
      expect(screen.getByText('Setup')).toBeInTheDocument();
    });
  });
});

// ---------------------------------------------------------------------------
// 7. Loading indicator shown while fetching
// ---------------------------------------------------------------------------
describe('loading state', () => {
  it('shows a loading indicator while the search is in flight', async () => {
    let resolveSearch!: (value: { query: string; results: import('../../api/search').SearchResultItem[]; wiki_summary: import('../../api/search').WikiSummaryItem[] }) => void;
    mockSearchWiki.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveSearch = resolve;
      }),
    );

    renderComponent({ wikiId: 'wiki-1' });
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'au' } });

    act(() => {
      jest.advanceTimersByTime(300);
    });

    // Loading spinner should appear while promise is pending
    await waitFor(() => {
      expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });

    // Resolve to clean up
    act(() => {
      resolveSearch({ query: 'au', results: [], wiki_summary: [] });
    });
  });
});

// ---------------------------------------------------------------------------
// 8. Empty state shown when no results
// ---------------------------------------------------------------------------
describe('empty state', () => {
  it('shows "No results found" when search returns empty', async () => {
    mockSearchWiki.mockResolvedValueOnce({ query: 'xyz', results: [], wiki_summary: [] });

    renderComponent({ wikiId: 'wiki-1' });
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'xyz' } });

    act(() => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => {
      expect(screen.getByText('No results found')).toBeInTheDocument();
    });
  });
});

// ---------------------------------------------------------------------------
// 9. onClose called on Escape
// ---------------------------------------------------------------------------
describe('close behaviour', () => {
  it('calls onClose when Escape key is pressed', async () => {
    const onClose = jest.fn();
    renderComponent({ open: true, onClose });

    fireEvent.keyDown(document, { key: 'Escape', code: 'Escape' });

    // MUI Dialog fires onClose via its own Escape handler — simulate via dialog close
    // MUI fires onClose on the Dialog element itself when Escape is pressed
    await waitFor(() => {
      // The dialog's backdrop or the dialog itself catches Escape
      const dialog = screen.getByRole('dialog');
      fireEvent.keyDown(dialog, { key: 'Escape', code: 'Escape' });
    });

    expect(onClose).toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// 10. Keyboard navigation: ArrowDown moves selection highlight
// ---------------------------------------------------------------------------
describe('keyboard navigation', () => {
  it('moves selection highlight down with ArrowDown', async () => {
    mockSearchWiki.mockResolvedValueOnce({
      query: 'test',
      results: [
        makeResult({ page_title: 'Overview' }),
        makeResult({ page_title: 'Architecture', wiki_id: 'wiki-1' }),
      ],
      wiki_summary: [],
    });

    renderComponent({ wikiId: 'wiki-1' });
    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: 'test' } });

    act(() => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => {
      expect(screen.getByText('Overview')).toBeInTheDocument();
    });

    // Press ArrowDown to move to second item
    fireEvent.keyDown(input, { key: 'ArrowDown', code: 'ArrowDown' });

    // The selected item state is tracked internally; pressing Enter on second item
    // should navigate to 'Architecture'
    mockNavigate.mockClear();
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith(
        `/wiki/wiki-1?page_title=${encodeURIComponent('Architecture')}`,
      );
    });
  });
});
