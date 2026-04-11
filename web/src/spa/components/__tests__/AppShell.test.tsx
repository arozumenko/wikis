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
