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
});
