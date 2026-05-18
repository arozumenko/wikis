/**
 * @jest-environment jsdom
 *
 * #210 — Project Ingestion tab tests.
 *
 * Covers:
 *  1. Ingestion tab renders the AddSourceWizard inline when URL is /project/:id/ingestion
 *  2. Successful ingestion links the wiki to the project then navigates to the wiki
 *  3. addWikiToProject failure surfaces a warning snack but still navigates
 *  4. 409 already-exists routes to the existing wiki without calling addWikiToProject
 */
import '@testing-library/jest-dom';
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';

// ---------------------------------------------------------------------------
// Mock project API
// ---------------------------------------------------------------------------
const mockAddWikiToProject = jest.fn();
const mockGetProject = jest.fn();
const mockListProjectWikis = jest.fn();

jest.mock('../../api/project', () => ({
  getProject: (...args: unknown[]) => mockGetProject(...args),
  listProjectWikis: (...args: unknown[]) => mockListProjectWikis(...args),
  updateProject: jest.fn(),
  deleteProject: jest.fn(),
  addWikiToProject: (...args: unknown[]) => mockAddWikiToProject(...args),
  removeWikiFromProject: jest.fn(),
}));

// ---------------------------------------------------------------------------
// Mock wiki API (generateWikiMultiSource used by the wizard)
// ---------------------------------------------------------------------------
const mockGenerateWikiMultiSource = jest.fn();
const mockScanSource = jest.fn();

jest.mock('../../api/wiki', () => {
  const actual = jest.requireActual('../../api/wiki');
  return {
    ...actual,
    generateWikiMultiSource: (...args: unknown[]) => mockGenerateWikiMultiSource(...args),
    scanSource: (...args: unknown[]) => mockScanSource(...args),
    listWikis: jest.fn().mockResolvedValue({ wikis: [] }),
  };
});

// ---------------------------------------------------------------------------
// Mock useConnections (used inside AddSourceWizard)
// ---------------------------------------------------------------------------
jest.mock('../../hooks/useConnections', () => ({
  useConnections: () => ({
    connections: [],
    atlassian: null,
    refreshAtlassianIfNeeded: jest.fn(),
  }),
}));

// ---------------------------------------------------------------------------
// Mock react-markdown + plugins (not available in jsdom)
// ---------------------------------------------------------------------------
jest.mock('react-markdown', () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));
jest.mock('remark-gfm', () => ({ __esModule: true, default: () => {} }));
jest.mock('rehype-highlight', () => ({ __esModule: true, default: () => {} }));

// ---------------------------------------------------------------------------
// Mock mermaid-dependent components (mermaid ships ESM only)
// ---------------------------------------------------------------------------
jest.mock('../../components/MermaidDiagram', () => ({
  MermaidDiagram: () => null,
}));
jest.mock('../../components/AnswerView', () => ({
  AnswerView: () => null,
}));
jest.mock('../../components/AnswerHeader', () => ({
  AnswerHeader: () => null,
}));
jest.mock('../../components/ToolCallPanel', () => ({
  ToolCallPanel: () => null,
}));
jest.mock('../../components/CodeMapTree', () => ({
  __esModule: true,
  default: () => null,
}));

// ---------------------------------------------------------------------------
// Mock RecomputeWidget — not relevant to these tests
// ---------------------------------------------------------------------------
jest.mock('../../components/RecomputeWidget', () => ({
  RecomputeWidget: () => null,
}));

// ---------------------------------------------------------------------------
// Mock AskBar — only present on the Overview tab, irrelevant here
// ---------------------------------------------------------------------------
jest.mock('../../components/AskBar', () => ({
  AskBar: () => null,
}));

// ---------------------------------------------------------------------------
// Capture navigation calls
// ---------------------------------------------------------------------------
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => {
  const actual = jest.requireActual<typeof import('react-router-dom')>('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

// ---------------------------------------------------------------------------
// ProjectContext provider (used by ProjectPage)
// ---------------------------------------------------------------------------
import { ProjectProvider } from '../../context/ProjectContext';
import { ProjectPage } from '../ProjectPage';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const PROJECT_ID = 'proj-123';

const mockProjectOwner = {
  id: PROJECT_ID,
  name: 'Test Project',
  description: null,
  visibility: 'personal' as const,
  owner_id: 'user-1',
  is_owner: true,
  created_at: '2025-01-01T00:00:00Z',
  wiki_count: 0,
};

const mockProjectNonOwner = {
  ...mockProjectOwner,
  is_owner: false,
};

function setup(
  initialPath = `/project/${PROJECT_ID}/ingestion`,
  project = mockProjectOwner,
) {
  mockGetProject.mockResolvedValue(project);
  mockListProjectWikis.mockResolvedValue({ wikis: [] });

  render(
    <MemoryRouter initialEntries={[initialPath]}>
      <Routes>
        <Route
          path="project/:projectId"
          element={
            <ProjectProvider>
              <ProjectPage />
            </ProjectProvider>
          }
        />
        <Route
          path="project/:projectId/:tab"
          element={
            <ProjectProvider>
              <ProjectPage />
            </ProjectProvider>
          }
        />
        {/* Catch-all so navigate() calls don't throw */}
        <Route path="*" element={<div data-testid="navigated-page" />} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  mockScanSource.mockResolvedValue({
    source_type: 'git',
    reachable: true,
    preview: {
      default_branch: null,
      resolved_branch: 'main',
      commit_hash: null,
      file_count: 5,
      top_paths: [],
      size_bytes: 0,
    },
    warnings: [],
  });
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Project Ingestion tab (#210)', () => {
  it('renders the AddSourceWizard inline when navigating to /ingestion', async () => {
    setup();
    // Wizard renders inline (no Dialog overlay) — the data-testid is on the root Box
    expect(await screen.findByTestId('add-source-wizard')).toBeInTheDocument();
    // The wizard's stepper is visible
    expect(screen.getByText('Connector')).toBeInTheDocument();
    expect(screen.getByText('Configure')).toBeInTheDocument();
  });

  it('success: links wiki to project then navigates to new wiki with generating params', async () => {
    const user = userEvent.setup();
    mockGenerateWikiMultiSource.mockResolvedValue({
      wiki_id: 'owner--repo--main',
      invocation_id: 'inv-42',
      status: 'generating',
      message: 'ok',
    });
    mockAddWikiToProject.mockResolvedValue({ ...mockProjectOwner, wiki_count: 1 });

    setup();
    await screen.findByTestId('add-source-wizard');

    // Walk through the wizard steps
    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('scan-success');
    await user.click(screen.getByTestId('wizard-next'));
    await user.click(screen.getByTestId('wizard-submit'));

    // addWikiToProject must be called with the project id and new wiki id
    await waitFor(() =>
      expect(mockAddWikiToProject).toHaveBeenCalledWith(PROJECT_ID, 'owner--repo--main'),
    );

    // navigate to the wiki with generating + invocation query params
    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith(
        expect.stringContaining('/wiki/owner--repo--main'),
      ),
    );
    const navCall = mockNavigate.mock.calls.find((args: string[]) =>
      (args[0] as string).includes('/wiki/owner--repo--main'),
    );
    expect(navCall?.[0]).toContain('generating=true');
    expect(navCall?.[0]).toContain('invocation=inv-42');
  });

  it('link failure shows a warning but still navigates to the new wiki', async () => {
    const user = userEvent.setup();
    mockGenerateWikiMultiSource.mockResolvedValue({
      wiki_id: 'owner--repo--main',
      invocation_id: null,
      status: 'generating',
      message: 'ok',
    });
    // Simulate a link failure
    mockAddWikiToProject.mockRejectedValue(new Error('Server error'));

    setup();
    await screen.findByTestId('add-source-wizard');

    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('scan-success');
    await user.click(screen.getByTestId('wizard-next'));
    await user.click(screen.getByTestId('wizard-submit'));

    // Should still navigate despite link error
    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith(
        expect.stringContaining('/wiki/owner--repo--main'),
      ),
    );

    // Warning message appears on the ingestion tab
    expect(
      await screen.findByText(/could not be automatically linked/i),
    ).toBeInTheDocument();
  });

  it('409 already-exists navigates to existing wiki without calling addWikiToProject', async () => {
    const user = userEvent.setup();
    const conflictErr = Object.assign(new Error('conflict'), {
      status: 409,
      body: { detail: { wiki_id: 'existing-wiki-id' } },
    });
    mockGenerateWikiMultiSource.mockRejectedValue(conflictErr);

    setup();
    await screen.findByTestId('add-source-wizard');

    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('scan-success');
    await user.click(screen.getByTestId('wizard-next'));
    await user.click(screen.getByTestId('wizard-submit'));

    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith('/wiki/existing-wiki-id'),
    );
    // Must NOT attempt to link — wiki already exists
    expect(mockAddWikiToProject).not.toHaveBeenCalled();
  });

  it('non-owner: direct link to /ingestion falls back to overview, no wizard rendered', async () => {
    // A non-owner following a direct link to /project/:id/ingestion must
    // land on the overview tab. The ingestion wizard must not render, and
    // the Ingestion tab must not appear in the tab bar.
    setup(`/project/${PROJECT_ID}/ingestion`, mockProjectNonOwner);

    // Wait for load to complete — the project name is a reliable marker
    expect(await screen.findByText('Test Project')).toBeInTheDocument();

    // Overview tab is active (tab label present)
    expect(screen.getByRole('tab', { name: 'Overview' })).toBeInTheDocument();

    // Ingestion tab is NOT shown for non-owners
    expect(screen.queryByTestId('tab-ingestion')).not.toBeInTheDocument();

    // The inline wizard must not render
    expect(screen.queryByTestId('add-source-wizard')).not.toBeInTheDocument();

    // Overview content IS visible — the empty-wikis alert is shown for non-owners
    expect(screen.getByText(/no wikis in this project yet/i)).toBeInTheDocument();
  });
});
