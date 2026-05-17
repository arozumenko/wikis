/**
 * @jest-environment jsdom
 */
import '@testing-library/jest-dom';
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { AppShell } from '../AppShell';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

// Mock QuickSearch so tests don't pull in the real dialog + API calls
jest.mock('../QuickSearch', () => ({
  QuickSearch: ({ open, onClose, wikiId, projectId }: {
    open: boolean;
    onClose: () => void;
    wikiId?: string;
    projectId?: string;
  }) => (
    open ? (
      <div
        data-testid="quick-search"
        data-wiki-id={wikiId ?? ''}
        data-project-id={projectId ?? ''}
      >
        <button onClick={onClose} aria-label="close quick search">Close</button>
      </div>
    ) : null
  ),
}));

// Mock useAuth — default to logged-in user
const mockSignOut = jest.fn();
jest.mock('../../hooks/useAuth', () => ({
  useAuth: () => ({
    user: { name: 'Test User', email: 'test@example.com', image: null },
    signOut: mockSignOut,
  }),
}));

// Mock refreshWiki
jest.mock('../../api/wiki', () => ({
  refreshWiki: jest.fn(),
}));

// Mock Outlet so we don't need nested routes
jest.mock('react-router-dom', () => {
  const actual = jest.requireActual<typeof import('react-router-dom')>('react-router-dom');
  return {
    ...actual,
    Outlet: () => <div data-testid="outlet" />,
  };
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderShell({
  path = '/',
  routePattern = '/',
}: {
  path?: string;
  routePattern?: string;
} = {}) {
  const defaultProps = {
    mode: 'light' as const,
    onToggleTheme: jest.fn(),
  };

  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path={routePattern} element={<AppShell {...defaultProps} />} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  jest.clearAllMocks();
});

// ---------------------------------------------------------------------------
// 1. Pressing Ctrl+K opens QuickSearch
// ---------------------------------------------------------------------------
describe('Ctrl+K / Cmd+K shortcut', () => {
  it('opens QuickSearch when Ctrl+K is pressed', () => {
    renderShell();
    expect(screen.queryByTestId('quick-search')).not.toBeInTheDocument();

    fireEvent.keyDown(window, { key: 'k', ctrlKey: true });

    expect(screen.getByTestId('quick-search')).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // 2. Pressing Cmd+K opens QuickSearch
  // ---------------------------------------------------------------------------
  it('opens QuickSearch when Cmd+K (metaKey) is pressed', () => {
    renderShell();
    expect(screen.queryByTestId('quick-search')).not.toBeInTheDocument();

    fireEvent.keyDown(window, { key: 'k', metaKey: true });

    expect(screen.getByTestId('quick-search')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// 3. QuickSearch closes when onClose is called
// ---------------------------------------------------------------------------
describe('QuickSearch close behaviour', () => {
  it('closes QuickSearch when onClose is called', () => {
    renderShell();

    // Open it
    fireEvent.keyDown(window, { key: 'k', ctrlKey: true });
    expect(screen.getByTestId('quick-search')).toBeInTheDocument();

    // Close via the button inside the mock
    fireEvent.click(screen.getByRole('button', { name: /close quick search/i }));

    expect(screen.queryByTestId('quick-search')).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// 4. Search icon button is visible in top bar
// ---------------------------------------------------------------------------
describe('Search icon button', () => {
  it('renders the Search icon button in the toolbar', () => {
    renderShell();
    expect(screen.getByRole('button', { name: /search/i })).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // 5. Clicking search icon opens QuickSearch
  // ---------------------------------------------------------------------------
  it('opens QuickSearch when the search icon button is clicked', () => {
    renderShell();
    expect(screen.queryByTestId('quick-search')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /search/i }));

    expect(screen.getByTestId('quick-search')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// 6. wikiId is passed to QuickSearch when on a wiki route
// ---------------------------------------------------------------------------
describe('wikiId threading', () => {
  it('passes wikiId to QuickSearch when on /wiki/:wikiId route', () => {
    renderShell({ path: '/wiki/my-wiki-123', routePattern: '/wiki/:wikiId' });

    // Open search
    fireEvent.keyDown(window, { key: 'k', ctrlKey: true });

    const qs = screen.getByTestId('quick-search');
    expect(qs).toHaveAttribute('data-wiki-id', 'my-wiki-123');
  });

  it('passes projectId to QuickSearch when on /project/:projectId route', () => {
    renderShell({ path: '/project/proj-456', routePattern: '/project/:projectId' });

    fireEvent.keyDown(window, { key: 'k', ctrlKey: true });

    const qs = screen.getByTestId('quick-search');
    expect(qs).toHaveAttribute('data-project-id', 'proj-456');
  });

  it('passes undefined wikiId/projectId when on the dashboard', () => {
    renderShell({ path: '/', routePattern: '/' });

    fireEvent.keyDown(window, { key: 'k', ctrlKey: true });

    const qs = screen.getByTestId('quick-search');
    expect(qs).toHaveAttribute('data-wiki-id', '');
    expect(qs).toHaveAttribute('data-project-id', '');
  });
});


// ---------------------------------------------------------------------------
// 4. Refresh-button flow (#172) — public vs private wiki
// ---------------------------------------------------------------------------

import { refreshWiki } from '../../api/wiki';

function renderShellWithRepo(repoContext: {
  wikiId?: string;
  repoUrl?: string;
  branch?: string;
  indexedAt?: string;
  commitHash?: string | null;
  requiresToken?: boolean;
}) {
  const defaultProps = {
    mode: 'light' as const,
    onToggleTheme: jest.fn(),
    repoContext,
  };
  return render(
    <MemoryRouter initialEntries={['/wiki/wiki-1']}>
      <Routes>
        <Route path="/wiki/:wikiId" element={<AppShell {...defaultProps} />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('Refresh button — PAT prompt for private wikis (#172)', () => {
  const mockedRefresh = refreshWiki as jest.MockedFunction<typeof refreshWiki>;

  beforeEach(() => {
    mockedRefresh.mockResolvedValue({
      wiki_id: 'wiki-1',
      invocation_id: 'inv-1',
      status: 'generating',
    } as Awaited<ReturnType<typeof refreshWiki>>);
  });

  it('public wiki — confirm dialog refreshes without prompting for a token', async () => {
    renderShellWithRepo({
      wikiId: 'wiki-1',
      repoUrl: 'https://github.com/owner/public-repo',
      requiresToken: false,
    });

    // Open confirm dialog
    fireEvent.click(screen.getByLabelText('Refresh wiki'));
    expect(screen.getByText('Refresh Wiki')).toBeInTheDocument();

    // Confirm — should call refreshWiki(wikiId) directly, no token
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));

    expect(mockedRefresh).toHaveBeenCalledWith('wiki-1', undefined);
    // Token dialog must NOT appear
    expect(
      screen.queryByText('GitHub access token required'),
    ).not.toBeInTheDocument();
  });

  it('private wiki — confirm dialog opens token modal, no refresh until token submitted', () => {
    renderShellWithRepo({
      wikiId: 'wiki-1',
      repoUrl: 'https://github.com/owner/private-repo',
      requiresToken: true,
    });

    fireEvent.click(screen.getByLabelText('Refresh wiki'));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));

    // Token modal opens
    expect(
      screen.getByText('GitHub access token required'),
    ).toBeInTheDocument();
    // Refresh API NOT called yet — waiting for the user's PAT
    expect(mockedRefresh).not.toHaveBeenCalled();
  });

  it('private wiki — submitting the PAT calls refreshWiki with the token', async () => {
    renderShellWithRepo({
      wikiId: 'wiki-1',
      repoUrl: 'https://github.com/owner/private-repo',
      requiresToken: true,
    });

    fireEvent.click(screen.getByLabelText('Refresh wiki'));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));

    const input = screen.getByLabelText('GitHub token') as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'ghp_test-token-123' } });

    // The token-modal's confirm button (distinct from the confirm
    // dialog's Refresh button which already closed).
    const modalRefreshBtn = screen
      .getAllByRole('button', { name: 'Refresh' })
      .find((b) => !b.hasAttribute('disabled'));
    fireEvent.click(modalRefreshBtn!);

    expect(mockedRefresh).toHaveBeenCalledWith('wiki-1', 'ghp_test-token-123');
  });

  it('private wiki — token modal Refresh button is disabled until a token is entered', () => {
    renderShellWithRepo({
      wikiId: 'wiki-1',
      repoUrl: 'https://github.com/owner/private-repo',
      requiresToken: true,
    });

    fireEvent.click(screen.getByLabelText('Refresh wiki'));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));

    // The modal's Refresh button is the only Refresh button still
    // visible at this point — the confirm-dialog one closed.
    const refreshButtons = screen.getAllByRole('button', { name: 'Refresh' });
    // All visible Refresh buttons must be disabled before any input.
    refreshButtons.forEach((btn) => expect(btn).toBeDisabled());
  });
});
