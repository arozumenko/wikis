/**
 * @jest-environment jsdom
 */
import '@testing-library/jest-dom';
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { GenerateForm } from '../GenerateForm';
import type { GenerateWikiMultiSourceRequest } from '../../api/wiki';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

// Mock useConnections — default: no connections
const mockRefreshAtlassianIfNeeded = jest.fn();
const mockUseConnections = jest.fn();

jest.mock('../../hooks/useConnections', () => ({
  useConnections: () => mockUseConnections(),
}));

// Helpers to set connection state
function noConnections() {
  mockUseConnections.mockReturnValue({
    connections: [],
    atlassian: null,
    refreshAtlassianIfNeeded: mockRefreshAtlassianIfNeeded,
  });
}

function withAtlassian(overrides: Record<string, unknown> = {}) {
  const atlassian = {
    provider: 'atlassian',
    access_token: 'at_test',
    refresh_token: 'rt_test',
    expires_at: Date.now() + 60 * 60 * 1000,
    cloud_id: 'cloud-abc',
    site_name: 'acme.atlassian.net',
    accessible_resources: [{ id: 'cloud-abc', name: 'acme', url: 'https://acme.atlassian.net', scopes: [] }],
    created_at: Date.now(),
    ...overrides,
  };
  mockUseConnections.mockReturnValue({
    connections: [atlassian],
    atlassian,
    refreshAtlassianIfNeeded: mockRefreshAtlassianIfNeeded,
  });
}

function withGitConnections() {
  const git = {
    provider: 'git',
    id: 'abc12345',
    repo_url: 'https://github.com/owner/repo',
    branch: 'main',
    pat: 'ghp_testtoken',
    label: 'My Repo',
    created_at: Date.now(),
  };
  mockUseConnections.mockReturnValue({
    connections: [git],
    atlassian: null,
    refreshAtlassianIfNeeded: mockRefreshAtlassianIfNeeded,
  });
}

// ---------------------------------------------------------------------------
// Render helper
// ---------------------------------------------------------------------------

function renderForm(onSubmit?: jest.Mock<Promise<void>, [GenerateWikiMultiSourceRequest]>) {
  const submit = onSubmit ?? jest.fn().mockResolvedValue(undefined);
  render(
    <MemoryRouter>
      <GenerateForm onSubmitMultiSource={submit} />
    </MemoryRouter>,
  );
  return { submit };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('GenerateForm — multi-source', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    noConnections();
  });

  // -------------------------------------------------------------------------
  // Default state
  // -------------------------------------------------------------------------

  it('renders Git tab active by default with empty fields', () => {
    renderForm();
    // Git tab should be active
    const gitTab = screen.getByTestId('tab-git');
    expect(gitTab).toHaveAttribute('aria-selected', 'true');
    // Repo URL input rendered
    expect(screen.getByTestId('git-repo-url')).toBeInTheDocument();
    // Submit button present
    expect(screen.getByTestId('generate-submit')).toBeInTheDocument();
  });

  it('submit button is disabled on Git tab until a valid repo URL is entered', async () => {
    // Updated behaviour (#276 Copilot review): submit is gated by ``isValid``
    // so users cannot trigger a no-op POST with an empty form.
    const user = userEvent.setup();
    renderForm();

    expect(screen.getByTestId('generate-submit')).toBeDisabled();

    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    expect(screen.getByTestId('generate-submit')).not.toBeDisabled();
  });

  // -------------------------------------------------------------------------
  // Confluence tab — no Atlassian connection
  // -------------------------------------------------------------------------

  it('switches to Confluence tab and shows connect-to-atlassian warning when no connection', async () => {
    const user = userEvent.setup();
    noConnections();
    renderForm();

    await user.click(screen.getByTestId('tab-confluence'));

    expect(await screen.findByTestId('atlassian-connect-warning')).toBeInTheDocument();
    expect(screen.getAllByText(/Connect to Atlassian/i).length).toBeGreaterThan(0);
  });

  it('disables submit button on Confluence tab when no Atlassian connection', async () => {
    const user = userEvent.setup();
    noConnections();
    renderForm();

    await user.click(screen.getByTestId('tab-confluence'));

    expect(screen.getByTestId('generate-submit')).toBeDisabled();
  });

  // -------------------------------------------------------------------------
  // Confluence tab — with Atlassian connection
  // -------------------------------------------------------------------------

  it('shows space keys input (not warning) when Atlassian connected on Confluence tab', async () => {
    const user = userEvent.setup();
    withAtlassian();
    renderForm();

    await user.click(screen.getByTestId('tab-confluence'));

    expect(screen.queryByTestId('atlassian-connect-warning')).not.toBeInTheDocument();
    expect(screen.getByTestId('space-keys-input')).toBeInTheDocument();
  });

  it('submit becomes enabled on Confluence tab when space key added', async () => {
    const user = userEvent.setup();
    withAtlassian();
    renderForm();

    await user.click(screen.getByTestId('tab-confluence'));

    const input = screen.getByTestId('space-keys-input');
    await user.type(input, 'ENG{Enter}');

    expect(screen.getByTestId('generate-submit')).not.toBeDisabled();
  });

  // -------------------------------------------------------------------------
  // Jira tab — no Atlassian connection
  // -------------------------------------------------------------------------

  it('shows connect-to-atlassian warning on Jira tab when no connection', async () => {
    const user = userEvent.setup();
    noConnections();
    renderForm();

    await user.click(screen.getByTestId('tab-jira'));

    expect(await screen.findByTestId('atlassian-connect-warning')).toBeInTheDocument();
    expect(screen.getByTestId('generate-submit')).toBeDisabled();
  });

  // -------------------------------------------------------------------------
  // Submit — Git request shape
  // -------------------------------------------------------------------------

  it('builds correct git request body on submit with no auth', async () => {
    const user = userEvent.setup();
    noConnections();
    const { submit } = renderForm();

    const urlInput = screen.getByTestId('git-repo-url');
    await user.clear(urlInput);
    await user.type(urlInput, 'https://github.com/owner/repo');

    // Branch already defaults to "main"
    await user.click(screen.getByTestId('generate-submit'));

    await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
    const body = submit.mock.calls[0][0] as GenerateWikiMultiSourceRequest;
    expect(body.source_type).toBe('git');
    expect(body.scope).toEqual({ repo_url: 'https://github.com/owner/repo', branch: 'main' });
    expect(body.auth).toEqual({ pat: null });
  });

  it('builds git request with pasted PAT', async () => {
    const user = userEvent.setup();
    noConnections();
    const { submit } = renderForm();

    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');

    // Switch auth to "paste"
    await user.click(screen.getByRole('combobox', { name: /authentication/i }));
    await user.click(await screen.findByText(/Paste token once/i));

    await user.type(screen.getByTestId('pasted-pat-input'), 'ghp_supersecret');

    await user.click(screen.getByTestId('generate-submit'));

    await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
    const body = submit.mock.calls[0][0] as GenerateWikiMultiSourceRequest;
    expect(body.source_type).toBe('git');
    expect((body.auth as { pat: string | null }).pat).toBe('ghp_supersecret');
  });

  it('submit is disabled when "Paste token once" is chosen with an empty token', async () => {
    // Copilot regression on #276: an empty pastedPat used to silently submit
    // as no-auth. Now the Submit button stays disabled until a token is
    // entered.
    const user = userEvent.setup();
    noConnections();
    renderForm();

    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByRole('combobox', { name: /authentication/i }));
    await user.click(await screen.findByText(/Paste token once/i));

    // Token field is empty — submit must be disabled.
    expect(screen.getByTestId('generate-submit')).toBeDisabled();

    // Once a token is provided, submit re-enables.
    await user.type(screen.getByTestId('pasted-pat-input'), 'ghp_realtoken');
    expect(screen.getByTestId('generate-submit')).not.toBeDisabled();
  });

  it('auth dropdown no longer offers stored PAT', async () => {
    // Regression: the stored-PAT path was removed because there is no
    // longer a UI to store PATs in the first place. The dropdown must
    // only offer "No auth" and "Paste token once".
    const user = userEvent.setup();
    withGitConnections();
    renderForm();

    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByRole('combobox', { name: /authentication/i }));

    const options = await screen.findAllByRole('option');
    const optionTexts = options.map((o) => o.textContent ?? '');
    expect(optionTexts.some((t) => /Use stored PAT/i.test(t))).toBe(false);
    expect(optionTexts.some((t) => /No auth/i.test(t))).toBe(true);
    expect(optionTexts.some((t) => /Paste token once/i.test(t))).toBe(true);
  });

  // -------------------------------------------------------------------------
  // Submit — Confluence request shape
  // -------------------------------------------------------------------------

  it('builds correct confluence request body on submit', async () => {
    const user = userEvent.setup();
    withAtlassian();
    mockRefreshAtlassianIfNeeded.mockResolvedValue({
      access_token: 'at_fresh',
      refresh_token: 'rt_fresh',
      expires_at: Date.now() + 3600_000,
      cloud_id: 'cloud-abc',
      site_name: 'acme.atlassian.net',
      accessible_resources: [{ id: 'cloud-abc', name: 'acme', url: 'https://acme.atlassian.net', scopes: [] }],
      created_at: Date.now(),
    });

    const { submit } = renderForm();

    await user.click(screen.getByTestId('tab-confluence'));

    const input = screen.getByTestId('space-keys-input');
    await user.type(input, 'ENG{Enter}');

    await user.click(screen.getByTestId('generate-submit'));

    await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
    const body = submit.mock.calls[0][0] as GenerateWikiMultiSourceRequest;
    expect(body.source_type).toBe('confluence');
    expect((body.scope as { space_keys: string[] }).space_keys).toContain('ENG');
    expect((body.scope as { base_url: string }).base_url).toBe('https://acme.atlassian.net');
    // auth should have access_token but no logging risk — just check shape
    expect('access_token' in body.auth).toBe(true);
  });

  // -------------------------------------------------------------------------
  // Submit — Jira request shape
  // -------------------------------------------------------------------------

  it('builds correct jira request body on submit', async () => {
    const user = userEvent.setup();
    withAtlassian();
    mockRefreshAtlassianIfNeeded.mockResolvedValue({
      access_token: 'at_fresh',
      refresh_token: 'rt_fresh',
      expires_at: Date.now() + 3600_000,
      cloud_id: 'cloud-abc',
      site_name: 'acme.atlassian.net',
      accessible_resources: [{ id: 'cloud-abc', name: 'acme', url: 'https://acme.atlassian.net', scopes: [] }],
      created_at: Date.now(),
    });

    const { submit } = renderForm();

    await user.click(screen.getByTestId('tab-jira'));

    // Clear default JQL and type new one
    const jqlInput = screen.getByTestId('jql-input');
    await user.clear(jqlInput);
    await user.type(jqlInput, 'project=ENG');

    await user.click(screen.getByTestId('generate-submit'));

    await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
    const body = submit.mock.calls[0][0] as GenerateWikiMultiSourceRequest;
    expect(body.source_type).toBe('jira');
    expect((body.scope as { jql: string }).jql).toBe('project=ENG');
    expect((body.scope as { base_url: string }).base_url).toBe('https://acme.atlassian.net');
  });

  // -------------------------------------------------------------------------
  // Atlassian token refresh
  // -------------------------------------------------------------------------

  it('calls refreshAtlassianIfNeeded before posting for Confluence', async () => {
    const user = userEvent.setup();
    withAtlassian();
    mockRefreshAtlassianIfNeeded.mockResolvedValue({
      access_token: 'at_fresh',
      refresh_token: 'rt_fresh',
      expires_at: Date.now() + 3600_000,
      cloud_id: 'cloud-abc',
      site_name: 'acme.atlassian.net',
      accessible_resources: [{ id: 'cloud-abc', name: 'acme', url: 'https://acme.atlassian.net', scopes: [] }],
      created_at: Date.now(),
    });

    const { submit } = renderForm();

    await user.click(screen.getByTestId('tab-confluence'));
    await user.type(screen.getByTestId('space-keys-input'), 'ENG{Enter}');
    await user.click(screen.getByTestId('generate-submit'));

    await waitFor(() => expect(mockRefreshAtlassianIfNeeded).toHaveBeenCalledTimes(1));
    expect(submit).toHaveBeenCalledTimes(1);
  });

  // -------------------------------------------------------------------------
  // Failed refresh
  // -------------------------------------------------------------------------

  it('shows error and does NOT call onSubmit when Atlassian refresh returns null', async () => {
    const user = userEvent.setup();
    withAtlassian();
    mockRefreshAtlassianIfNeeded.mockResolvedValue(null);

    const { submit } = renderForm();

    await user.click(screen.getByTestId('tab-confluence'));
    await user.type(screen.getByTestId('space-keys-input'), 'ENG{Enter}');
    await user.click(screen.getByTestId('generate-submit'));

    await waitFor(() =>
      expect(screen.getByTestId('submit-error')).toHaveTextContent(
        /Atlassian connection lost/i,
      ),
    );
    expect(submit).not.toHaveBeenCalled();
  });
});
