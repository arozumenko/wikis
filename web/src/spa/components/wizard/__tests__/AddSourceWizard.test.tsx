/**
 * @jest-environment jsdom
 *
 * #208 — AddSourceWizard exercises the four-step flow end-to-end.
 *
 * Covers: connector picker advances to Configure, Configure validates,
 * Scan calls the new /sources/scan endpoint and renders the preview,
 * Skip-preview bypasses Scan, and final submit dispatches to
 * generateWikiMultiSource. 409 conflict path routes to onAlreadyExists.
 */
import '@testing-library/jest-dom';
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';

// Mock the API module so wizard interactions hit our fakes.
jest.mock('../../../api/wiki', () => {
  const actual = jest.requireActual('../../../api/wiki');
  return {
    ...actual,
    scanSource: jest.fn(),
    generateWikiMultiSource: jest.fn(),
  };
});

// Mock useConnections so we can drive the Atlassian / git PAT branches.
const mockRefreshAtlassianIfNeeded = jest.fn();
const mockUseConnections = jest.fn();
jest.mock('../../../hooks/useConnections', () => ({
  useConnections: () => mockUseConnections(),
}));

import { AddSourceWizard } from '../AddSourceWizard';
import { scanSource, generateWikiMultiSource } from '../../../api/wiki';
import { ApiError } from '../../../api/client';

const mockScanSource = scanSource as jest.MockedFunction<typeof scanSource>;
const mockGenerate = generateWikiMultiSource as jest.MockedFunction<
  typeof generateWikiMultiSource
>;

function noConnections() {
  mockUseConnections.mockReturnValue({
    connections: [],
    atlassian: null,
    refreshAtlassianIfNeeded: mockRefreshAtlassianIfNeeded,
  });
}

function renderWizard(overrides: Partial<React.ComponentProps<typeof AddSourceWizard>> = {}) {
  const props: React.ComponentProps<typeof AddSourceWizard> = {
    open: true,
    onClose: jest.fn(),
    onSuccess: jest.fn(),
    ...overrides,
  };
  render(
    <MemoryRouter>
      <AddSourceWizard {...props} />
    </MemoryRouter>,
  );
  return props;
}

beforeEach(() => {
  jest.clearAllMocks();
  noConnections();
  mockScanSource.mockReset();
  mockGenerate.mockReset();
});

describe('AddSourceWizard', () => {
  it('starts on the Connector step with Git selected by default', () => {
    renderWizard();
    // Stepper labels all rendered.
    expect(screen.getByText('Connector')).toBeInTheDocument();
    expect(screen.getByText('Configure')).toBeInTheDocument();
    expect(screen.getByText('Scan')).toBeInTheDocument();
    expect(screen.getByText('Confirm')).toBeInTheDocument();
    // Step 1 cards are present.
    expect(screen.getByTestId('connector-git')).toBeInTheDocument();
    expect(screen.getByTestId('connector-confluence')).toBeInTheDocument();
    expect(screen.getByTestId('connector-jira')).toBeInTheDocument();
  });

  it('clicking a connector auto-advances to Configure', async () => {
    const user = userEvent.setup();
    renderWizard();
    await user.click(screen.getByTestId('connector-git'));
    // Step 2's Git URL field appears.
    expect(await screen.findByTestId('git-repo-url')).toBeInTheDocument();
  });

  it('Next is disabled until the URL passes validation', async () => {
    const user = userEvent.setup();
    renderWizard();
    await user.click(screen.getByTestId('connector-git'));
    const next = await screen.findByTestId('wizard-next');
    expect(next).toBeDisabled();
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    expect(next).toBeEnabled();
  });

  it('Scan step renders preview from /sources/scan', async () => {
    const user = userEvent.setup();
    mockScanSource.mockResolvedValue({
      source_type: 'git',
      reachable: true,
      preview: {
        default_branch: null,
        resolved_branch: 'main',
        commit_hash: 'abc1234deadbeef',
        file_count: 42,
        top_paths: ['README.md', 'src/'],
        size_bytes: 12345,
      },
      warnings: [],
    });
    renderWizard();
    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByTestId('wizard-next'));

    expect(await screen.findByTestId('scan-success')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument(); // file count
    expect(screen.getByText('main')).toBeInTheDocument();
    expect(mockScanSource).toHaveBeenCalledTimes(1);
    expect(mockScanSource).toHaveBeenCalledWith(
      expect.objectContaining({
        source_type: 'git',
        scope: { repo_url: 'https://github.com/owner/repo', branch: 'main' },
        auth: { pat: null },
      }),
    );
  });

  it('Skip preview goes to Confirm without scanning', async () => {
    const user = userEvent.setup();
    mockScanSource.mockImplementation(() => new Promise(() => {})); // never resolves
    renderWizard();
    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByTestId('wizard-next'));
    await user.click(screen.getByTestId('wizard-skip-scan'));
    expect(await screen.findByTestId('confirm-scan-skipped')).toBeInTheDocument();
  });

  it('submit dispatches POST /wikis with the multi-source request shape', async () => {
    const user = userEvent.setup();
    mockScanSource.mockResolvedValue({
      source_type: 'git',
      reachable: true,
      preview: {
        default_branch: null,
        resolved_branch: 'main',
        commit_hash: null,
        file_count: 1,
        top_paths: [],
        size_bytes: 0,
      },
      warnings: [],
    });
    mockGenerate.mockResolvedValue({
      wiki_id: 'owner--repo--main',
      invocation_id: 'inv-1',
      status: 'generating',
      message: 'ok',
    });
    const onSuccess = jest.fn();
    renderWizard({ onSuccess });
    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('scan-success');
    await user.click(screen.getByTestId('wizard-next'));
    await user.click(screen.getByTestId('wizard-submit'));

    await waitFor(() => expect(mockGenerate).toHaveBeenCalledTimes(1));
    expect(mockGenerate).toHaveBeenCalledWith(
      expect.objectContaining({
        source_type: 'git',
        scope: { repo_url: 'https://github.com/owner/repo', branch: 'main' },
        auth: { pat: null },
        structure_planner: 'agentic',
      }),
    );
    expect(onSuccess).toHaveBeenCalledWith(
      expect.objectContaining({ wiki_id: 'owner--repo--main', invocation_id: 'inv-1' }),
    );
  });

  it('409 conflict routes to onAlreadyExists with the existing wiki_id', async () => {
    const user = userEvent.setup();
    mockScanSource.mockResolvedValue({
      source_type: 'git',
      reachable: true,
      preview: {
        default_branch: null,
        resolved_branch: 'main',
        commit_hash: null,
        file_count: 1,
        top_paths: [],
        size_bytes: 0,
      },
      warnings: [],
    });
    const conflictErr = Object.assign(new Error('exists'), {
      status: 409,
      body: { detail: { wiki_id: 'owner--repo--main' } },
    });
    mockGenerate.mockRejectedValue(conflictErr);
    const onAlreadyExists = jest.fn();
    renderWizard({ onAlreadyExists });
    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('scan-success');
    await user.click(screen.getByTestId('wizard-next'));
    await user.click(screen.getByTestId('wizard-submit'));

    await waitFor(() =>
      expect(onAlreadyExists).toHaveBeenCalledWith('owner--repo--main'),
    );
  });

  // -------------------------------------------------------------------
  // Round-2 fixes (Rio review on PR #216)
  // -------------------------------------------------------------------

  it('Back is enabled on the Confirm step (C1)', async () => {
    const user = userEvent.setup();
    mockScanSource.mockResolvedValue({
      source_type: 'git',
      reachable: true,
      preview: {
        default_branch: null,
        resolved_branch: 'main',
        commit_hash: null,
        file_count: 1,
        top_paths: [],
        size_bytes: 0,
      },
      warnings: [],
    });
    renderWizard();
    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('scan-success');
    await user.click(screen.getByTestId('wizard-next'));
    // We're on Confirm now — Back must be present and enabled.
    const back = screen.getByTestId('wizard-back');
    expect(back).toBeEnabled();
    await user.click(back);
    // We should be back on Scan (not Configure), and the cached scan
    // result must render without re-hitting the network (C7 / Rio).
    expect(await screen.findByTestId('scan-success')).toBeInTheDocument();
    expect(mockScanSource).toHaveBeenCalledTimes(1);
  });

  it('Dirty wizard prompts before closing (C5 — AC)', async () => {
    const user = userEvent.setup();
    const onClose = jest.fn();
    renderWizard({ onClose });
    // Pick a connector and type something so the wizard is dirty.
    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    // Hit Cancel.
    await user.click(screen.getByText('Cancel'));
    // onClose must NOT have fired yet — the confirm-discard dialog must appear.
    expect(onClose).not.toHaveBeenCalled();
    expect(await screen.findByTestId('wizard-discard-confirm')).toBeInTheDocument();
    // Keep editing dismisses the confirm and does not close.
    await user.click(screen.getByTestId('wizard-discard-cancel'));
    expect(onClose).not.toHaveBeenCalled();
    // Cancel again, then confirm discard — now onClose fires.
    await user.click(screen.getByText('Cancel'));
    await user.click(screen.getByTestId('wizard-discard-confirm-button'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('Connector-pick-only is not dirty — close without prompt', async () => {
    const user = userEvent.setup();
    const onClose = jest.fn();
    renderWizard({ onClose });
    // The wizard auto-opens on Git; picking the same connector and
    // closing should not trigger the discard prompt.
    await user.click(screen.getByText('Cancel'));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(screen.queryByTestId('wizard-discard-confirm')).not.toBeInTheDocument();
  });

  it('Skip-preview during an in-flight scan does not overwrite scanSkipped (C2)', async () => {
    const user = userEvent.setup();
    type Resolver = (value: import('../../../api/wiki').ScanResponse) => void;
    // Cast around TS's CFA — it narrows the captured-in-Promise variable
    // back to ``null`` even though the executor assigns it synchronously.
    let resolveScan: Resolver | null = null;
    mockScanSource.mockImplementation(
      () =>
        new Promise<import('../../../api/wiki').ScanResponse>((resolve) => {
          resolveScan = resolve as Resolver;
        }),
    );
    renderWizard();
    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByTestId('wizard-next'));
    // Skip while the scan is mid-flight.
    await user.click(screen.getByTestId('wizard-skip-scan'));
    expect(await screen.findByTestId('confirm-scan-skipped')).toBeInTheDocument();
    // Now resolve the late scan — Confirm must STILL show the skipped notice,
    // not the success stats.
    (resolveScan as Resolver | null)?.({
      source_type: 'git',
      reachable: true,
      preview: {
        default_branch: null,
        resolved_branch: 'main',
        commit_hash: null,
        file_count: 42,
        top_paths: [],
        size_bytes: 0,
      },
      warnings: [],
    });
    // Give React a tick to apply any late state update.
    await new Promise((r) => setTimeout(r, 0));
    expect(screen.getByTestId('confirm-scan-skipped')).toBeInTheDocument();
    expect(screen.queryByTestId('confirm-scan-stats')).not.toBeInTheDocument();
  });

  it('Scan 501 (unsupported connector) lets the user continue to Confirm', async () => {
    // The wizard treats any 501 from /sources/scan as "preview not
    // available for this source yet" and surfaces a skip-to-Confirm
    // affordance. Exercised here via Git + a forced 501 so we focus on
    // the response handling rather than the Confluence chip-entry UX.
    const user = userEvent.setup();
    mockScanSource.mockRejectedValue(
      new ApiError(501, { detail: 'Scan not implemented' }),
    );
    renderWizard();
    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByTestId('wizard-next'));
    expect(await screen.findByTestId('scan-unsupported')).toBeInTheDocument();
    // Skip-to-Confirm is still reachable via Next on the Scan step.
    await user.click(screen.getByTestId('wizard-next'));
    expect(await screen.findByTestId('step-confirm')).toBeInTheDocument();
  });

  // -------------------------------------------------------------------
  // #217 — auth presence in scan cache hash
  // -------------------------------------------------------------------

  it('#217: toggling from no-PAT to paste-PAT causes a cache miss and re-scan', async () => {
    // First scan: no PAT → backend returns reachable but no preview.
    // Then user switches to paste mode and adds a PAT.
    // Re-entering Scan step must call scanSource again (cache miss) because
    // auth presence changed (null → non-null), invalidating the hash (#217).
    const user = userEvent.setup();

    // First scan resolves with reachable=true but no preview.
    mockScanSource.mockResolvedValueOnce({
      source_type: 'git',
      reachable: true,
      preview: null,
      warnings: [],
    });
    // Second scan (after adding a PAT) resolves with a full preview.
    mockScanSource.mockResolvedValueOnce({
      source_type: 'git',
      reachable: true,
      preview: {
        default_branch: null,
        resolved_branch: 'main',
        commit_hash: null,
        file_count: 10,
        top_paths: [],
        size_bytes: 0,
      },
      warnings: [],
    });

    renderWizard();
    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/private-repo');
    // Advance to Scan with no PAT — first scan fires.
    await user.click(screen.getByTestId('wizard-next'));
    // Wait for the first scan result (reachable, no preview).
    await screen.findByTestId('scan-no-preview');
    expect(mockScanSource).toHaveBeenCalledTimes(1);

    // Go back to Configure and switch to paste-PAT mode.
    await user.click(screen.getByTestId('wizard-back'));
    // MUI Select renders a hidden native input (pointer-events: none) for the
    // data-testid. Click the visible combobox div (role="combobox") instead.
    const patSourceCombobox = screen.getAllByRole('combobox')[0];
    await user.click(patSourceCombobox);
    await user.click(await screen.findByText('Paste token once (not stored)'));
    // Enter a PAT value.
    await user.type(screen.getByTestId('pasted-pat-input'), 'ghp_secrettoken');

    // Re-enter Scan step — auth presence changed (null → non-null),
    // so the cache hash differs and scanSource must be called again.
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('scan-success');
    expect(mockScanSource).toHaveBeenCalledTimes(2);
  });

  // -------------------------------------------------------------------
  // #221 — Confluence preview rendering
  // -------------------------------------------------------------------

  it('#221: Confluence preview renders spaces list and total_pages', async () => {
    const user = userEvent.setup();

    // Wire up Atlassian connection so the wizard can advance past Configure.
    mockUseConnections.mockReturnValue({
      connections: [],
      atlassian: {
        access_token: 'tok',
        refresh_token: null,
        site_name: 'mysite.atlassian.net',
        accessible_resources: [{ id: 'r1', url: 'https://mysite.atlassian.net', name: 'My Site' }],
      },
      refreshAtlassianIfNeeded: mockRefreshAtlassianIfNeeded,
    });

    mockScanSource.mockResolvedValue({
      source_type: 'confluence',
      reachable: true,
      preview: {
        spaces: [
          { key: 'ENG', name: 'Engineering', page_count: 120 },
          { key: 'DEV', name: 'DevOps', page_count: null },
        ],
        total_pages: null,
      },
      warnings: [],
    });

    renderWizard();
    // Pick Confluence connector.
    await user.click(screen.getByTestId('connector-confluence'));
    // Configure step — add at least one space key via the autocomplete input.
    const spaceInput = await screen.findByTestId('space-keys-input');
    await user.type(spaceInput, 'ENG');
    await user.keyboard('{Enter}');
    // Advance to Scan.
    await user.click(screen.getByTestId('wizard-next'));

    // Scan success panel for Confluence.
    expect(await screen.findByTestId('scan-success')).toBeInTheDocument();

    // Space chips: ENG with page count, DEV with "?".
    expect(screen.getByTestId('confluence-space-ENG')).toBeInTheDocument();
    expect(screen.getByText('Engineering (120 pages)')).toBeInTheDocument();
    expect(screen.getByTestId('confluence-space-DEV')).toBeInTheDocument();
    expect(screen.getByText('DevOps (?)')).toBeInTheDocument();

    // total_pages is null → should show '?'
    expect(screen.getByText('?')).toBeInTheDocument();
  });

  // -------------------------------------------------------------------
  // #221 — Jira preview rendering
  // -------------------------------------------------------------------

  it('#221: Jira preview renders matching_issues, sample keys, and JQL-validated indicator', async () => {
    const user = userEvent.setup();

    mockUseConnections.mockReturnValue({
      connections: [],
      atlassian: {
        access_token: 'tok',
        refresh_token: null,
        site_name: 'mysite.atlassian.net',
        accessible_resources: [{ id: 'r1', url: 'https://mysite.atlassian.net', name: 'My Site' }],
      },
      refreshAtlassianIfNeeded: mockRefreshAtlassianIfNeeded,
    });

    mockScanSource.mockResolvedValue({
      source_type: 'jira',
      reachable: true,
      preview: {
        matching_issues: 47,
        sample_issue_keys: ['PROJ-1', 'PROJ-2', 'PROJ-3'],
        jql_validated: true,
      },
      warnings: [],
    });

    renderWizard();
    await user.click(screen.getByTestId('connector-jira'));
    // JiraConfigure has a JQL text area pre-filled — just advance.
    await user.click(screen.getByTestId('wizard-next'));

    expect(await screen.findByTestId('scan-success')).toBeInTheDocument();

    // Matching issues count.
    expect(screen.getByText('47')).toBeInTheDocument();
    // JQL validated indicator.
    expect(screen.getByText('✓ Yes')).toBeInTheDocument();
    // Sample issue key chips.
    expect(screen.getByTestId('jira-issue-PROJ-1')).toBeInTheDocument();
    expect(screen.getByTestId('jira-issue-PROJ-2')).toBeInTheDocument();
    expect(screen.getByTestId('jira-issue-PROJ-3')).toBeInTheDocument();
  });

  // -------------------------------------------------------------------
  // PR #223 round-2 — Confirm-step buildPreviewSummary() for Confluence
  // and Jira (Rio review: untested branches).
  // -------------------------------------------------------------------

  it('#223: Confirm step shows correct summary string for Confluence preview (total_pages known)', async () => {
    // Drive wizard end-to-end: Confluence connector → Configure (add space
    // key) → Scan (mocked to return Confluence preview) → Confirm.
    // Assert the confirm-scan-stats Alert text matches buildPreviewSummary().
    const user = userEvent.setup();

    mockUseConnections.mockReturnValue({
      connections: [],
      atlassian: {
        access_token: 'tok',
        refresh_token: null,
        site_name: 'mysite.atlassian.net',
        accessible_resources: [{ id: 'r1', url: 'https://mysite.atlassian.net', name: 'My Site' }],
      },
      refreshAtlassianIfNeeded: mockRefreshAtlassianIfNeeded,
    });

    mockScanSource.mockResolvedValue({
      source_type: 'confluence',
      reachable: true,
      preview: {
        spaces: [
          { key: 'ENG', name: 'Engineering', page_count: 80 },
          { key: 'OPS', name: 'Ops', page_count: 40 },
        ],
        total_pages: 120,
      },
      warnings: [],
    });

    renderWizard();
    await user.click(screen.getByTestId('connector-confluence'));
    const spaceInput = await screen.findByTestId('space-keys-input');
    await user.type(spaceInput, 'ENG');
    await user.keyboard('{Enter}');
    // Advance through Scan.
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('scan-success');
    // Advance to Confirm.
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('step-confirm');

    // buildPreviewSummary for confluence with 2 spaces and total_pages=120:
    // "Preview: 2 spaces, 120 pages."
    const statsAlert = screen.getByTestId('confirm-scan-stats');
    expect(statsAlert).toHaveTextContent('Preview: 2 spaces, 120 pages.');
  });

  it('#223: Confirm step shows "?" for Confluence preview when total_pages is null', async () => {
    const user = userEvent.setup();

    mockUseConnections.mockReturnValue({
      connections: [],
      atlassian: {
        access_token: 'tok',
        refresh_token: null,
        site_name: 'mysite.atlassian.net',
        accessible_resources: [{ id: 'r1', url: 'https://mysite.atlassian.net', name: 'My Site' }],
      },
      refreshAtlassianIfNeeded: mockRefreshAtlassianIfNeeded,
    });

    mockScanSource.mockResolvedValue({
      source_type: 'confluence',
      reachable: true,
      preview: {
        spaces: [
          { key: 'ENG', name: 'Engineering', page_count: null },
          { key: 'OPS', name: 'Ops', page_count: null },
        ],
        // total_pages unknown — backend didn't enumerate everything.
        total_pages: null,
      },
      warnings: [],
    });

    renderWizard();
    await user.click(screen.getByTestId('connector-confluence'));
    const spaceInput = await screen.findByTestId('space-keys-input');
    await user.type(spaceInput, 'ENG');
    await user.keyboard('{Enter}');
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('scan-success');
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('step-confirm');

    // buildPreviewSummary for confluence with 2 spaces and total_pages=null:
    // "Preview: 2 spaces, ? pages."
    const statsAlert = screen.getByTestId('confirm-scan-stats');
    expect(statsAlert).toHaveTextContent('Preview: 2 spaces, ? pages.');
  });

  it('#223: Confirm step shows correct summary string for Jira preview (JQL validated)', async () => {
    const user = userEvent.setup();

    mockUseConnections.mockReturnValue({
      connections: [],
      atlassian: {
        access_token: 'tok',
        refresh_token: null,
        site_name: 'mysite.atlassian.net',
        accessible_resources: [{ id: 'r1', url: 'https://mysite.atlassian.net', name: 'My Site' }],
      },
      refreshAtlassianIfNeeded: mockRefreshAtlassianIfNeeded,
    });

    mockScanSource.mockResolvedValue({
      source_type: 'jira',
      reachable: true,
      preview: {
        matching_issues: 47,
        sample_issue_keys: ['PROJ-1', 'PROJ-2'],
        jql_validated: true,
      },
      warnings: [],
    });

    renderWizard();
    await user.click(screen.getByTestId('connector-jira'));
    // JiraConfigure has a default JQL — just advance.
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('scan-success');
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('step-confirm');

    // buildPreviewSummary for jira with matching_issues=47, jql_validated=true:
    // "Preview: 47 matching issues, JQL validated."
    const statsAlert = screen.getByTestId('confirm-scan-stats');
    expect(statsAlert).toHaveTextContent('Preview: 47 matching issues, JQL validated.');
  });
});
