/**
 * @jest-environment jsdom
 *
 * Tests for the inline AtlassianConnect component plus the
 * ConfluenceConfigure / JiraConfigure integration.
 *
 * #28 expanded the auth surface to two tabs: OAuth (the prior single-button
 * flow) and API token (email + token). These tests cover both.
 */
import '@testing-library/jest-dom';
import React from 'react';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const mockStartOAuth = jest.fn();
jest.mock('../../../../lib/atlassian-oauth', () => ({
  startAtlassianOAuth: () => mockStartOAuth(),
}));

import { AtlassianConnect } from '../AtlassianConnect';
import type { AtlassianBasicAuthFormState } from '../../types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const EMPTY_BASIC: AtlassianBasicAuthFormState = {
  siteUrl: '',
  email: '',
  apiToken: '',
};

function defaultProps(overrides: Partial<React.ComponentProps<typeof AtlassianConnect>> = {}) {
  return {
    mode: 'oauth' as const,
    onModeChange: jest.fn(),
    basic: EMPTY_BASIC,
    onBasicChange: jest.fn(),
    onConnected: jest.fn(),
    ...overrides,
  };
}

beforeEach(() => {
  jest.clearAllMocks();
  mockStartOAuth.mockResolvedValue(undefined);
});

// ---------------------------------------------------------------------------
// OAuth tab
// ---------------------------------------------------------------------------

describe('AtlassianConnect — OAuth tab', () => {
  it('renders the Connect to Atlassian button by default (oauth mode)', () => {
    render(<AtlassianConnect {...defaultProps()} />);
    expect(screen.getByTestId('atlassian-connect-button')).toBeInTheDocument();
    expect(screen.getByText('Connect to Atlassian')).toBeInTheDocument();
  });

  it('clicking the button calls startAtlassianOAuth', async () => {
    const user = userEvent.setup();
    render(<AtlassianConnect {...defaultProps()} />);
    await user.click(screen.getByTestId('atlassian-connect-button'));
    expect(mockStartOAuth).toHaveBeenCalledTimes(1);
  });

  it('shows a spinner while waiting for the popup', async () => {
    const user = userEvent.setup();
    mockStartOAuth.mockImplementation(() => new Promise(() => {}));
    render(<AtlassianConnect {...defaultProps()} />);
    await user.click(screen.getByTestId('atlassian-connect-button'));
    expect(await screen.findByRole('progressbar')).toBeInTheDocument();
    expect(screen.queryByTestId('atlassian-connect-button')).not.toBeInTheDocument();
  });

  it('postMessage wikis-oauth-success calls onConnected after delay', async () => {
    jest.useFakeTimers();
    const onConnected = jest.fn();
    render(<AtlassianConnect {...defaultProps({ onConnected })} />);

    act(() => {
      window.dispatchEvent(
        new MessageEvent('message', {
          data: { type: 'wikis-oauth-success', provider: 'atlassian' },
          origin: window.location.origin,
        }),
      );
    });

    expect(onConnected).not.toHaveBeenCalled();

    act(() => {
      jest.advanceTimersByTime(700);
    });

    expect(onConnected).toHaveBeenCalledTimes(1);
    jest.useRealTimers();
  });

  it('ignores postMessages from different origins', () => {
    const onConnected = jest.fn();
    render(<AtlassianConnect {...defaultProps({ onConnected })} />);

    act(() => {
      window.dispatchEvent(
        new MessageEvent('message', {
          data: { type: 'wikis-oauth-success', provider: 'atlassian' },
          origin: 'https://evil.example.com',
        }),
      );
    });

    expect(onConnected).not.toHaveBeenCalled();
  });

  it('shows an error alert when startAtlassianOAuth throws', async () => {
    const user = userEvent.setup();
    mockStartOAuth.mockRejectedValue(new Error('Popup blocked'));
    render(<AtlassianConnect {...defaultProps()} />);
    await user.click(screen.getByTestId('atlassian-connect-button'));
    expect(await screen.findByRole('alert')).toHaveTextContent('Popup blocked');
  });
});

// ---------------------------------------------------------------------------
// API-token tab (#28)
// ---------------------------------------------------------------------------

describe('AtlassianConnect — API-token tab', () => {
  it('renders the API-token fields when mode is api_token', () => {
    render(<AtlassianConnect {...defaultProps({ mode: 'api_token' })} />);
    expect(screen.getByTestId('atlassian-basic-site-url')).toBeInTheDocument();
    expect(screen.getByTestId('atlassian-basic-email')).toBeInTheDocument();
    expect(screen.getByTestId('atlassian-basic-api-token')).toBeInTheDocument();
    // No OAuth button in API-token mode
    expect(screen.queryByTestId('atlassian-connect-button')).not.toBeInTheDocument();
  });

  it('clicking the API-token tab fires onModeChange', async () => {
    const user = userEvent.setup();
    const onModeChange = jest.fn();
    render(<AtlassianConnect {...defaultProps({ onModeChange })} />);
    await user.click(screen.getByTestId('atlassian-tab-api-token'));
    expect(onModeChange).toHaveBeenCalledWith('api_token');
  });

  it('typing in the email field bubbles to onBasicChange', async () => {
    const user = userEvent.setup();
    const onBasicChange = jest.fn();
    render(
      <AtlassianConnect
        {...defaultProps({ mode: 'api_token', onBasicChange })}
      />,
    );
    await user.type(screen.getByTestId('atlassian-basic-email'), 'a');
    expect(onBasicChange).toHaveBeenLastCalledWith({
      siteUrl: '',
      email: 'a',
      apiToken: '',
    });
  });
});

// ---------------------------------------------------------------------------
// ConfluenceConfigure / JiraConfigure integration
// ---------------------------------------------------------------------------

import { MemoryRouter } from 'react-router-dom';

const mockUseConnections = jest.fn();
jest.mock('../../../../hooks/useConnections', () => ({
  useConnections: () => mockUseConnections(),
}));

import { ConfluenceConfigure } from '../ConfluenceConfigure';
import { JiraConfigure } from '../JiraConfigure';

function noAtlassian() {
  mockUseConnections.mockReturnValue({
    connections: [],
    atlassian: null,
    saveAtlassian: jest.fn(),
    refreshAtlassianIfNeeded: jest.fn(),
  });
}

function withAtlassian() {
  const atlassian = {
    provider: 'atlassian',
    access_token: 'at',
    refresh_token: 'rt',
    expires_at: Date.now() + 3600_000,
    cloud_id: 'cloud-1',
    site_name: 'acme.atlassian.net',
    accessible_resources: [
      { id: 'cloud-1', name: 'acme', url: 'https://acme.atlassian.net', scopes: [] },
    ],
    created_at: Date.now(),
  };
  mockUseConnections.mockReturnValue({
    connections: [atlassian],
    atlassian,
    saveAtlassian: jest.fn(),
    refreshAtlassianIfNeeded: jest.fn(),
  });
}

function confluenceProps() {
  return {
    data: { space_keys: [] },
    onChange: jest.fn(),
    spaceKeysError: null,
    authMode: 'oauth' as const,
    onAuthModeChange: jest.fn(),
    basicAuth: EMPTY_BASIC,
    onBasicAuthChange: jest.fn(),
  };
}

function jiraProps() {
  return {
    data: { jql: 'ORDER BY created DESC' },
    onChange: jest.fn(),
    jqlError: null,
    authMode: 'oauth' as const,
    onAuthModeChange: jest.fn(),
    basicAuth: EMPTY_BASIC,
    onBasicAuthChange: jest.fn(),
  };
}

describe('ConfluenceConfigure — AtlassianConnect integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    noAtlassian();
  });

  it('renders the OAuth Connect button when no Atlassian session and mode is oauth', () => {
    render(
      <MemoryRouter>
        <ConfluenceConfigure {...confluenceProps()} />
      </MemoryRouter>,
    );
    expect(screen.getByTestId('atlassian-connect-button')).toBeInTheDocument();
    expect(screen.queryByTestId('space-keys-input')).not.toBeInTheDocument();
  });

  it('renders the space-keys input once OAuth is connected', () => {
    withAtlassian();
    render(
      <MemoryRouter>
        <ConfluenceConfigure {...confluenceProps()} />
      </MemoryRouter>,
    );
    expect(screen.getByTestId('space-keys-input')).toBeInTheDocument();
  });

  it('switching to API-token mode reveals the basic-auth fields', () => {
    render(
      <MemoryRouter>
        <ConfluenceConfigure
          {...confluenceProps()}
          authMode="api_token"
        />
      </MemoryRouter>,
    );
    expect(screen.getByTestId('atlassian-basic-site-url')).toBeInTheDocument();
    expect(screen.getByTestId('atlassian-basic-email')).toBeInTheDocument();
    expect(screen.getByTestId('atlassian-basic-api-token')).toBeInTheDocument();
    // No OAuth button when in api_token mode
    expect(screen.queryByTestId('atlassian-connect-button')).not.toBeInTheDocument();
  });

  it('api_token mode with all 3 fields filled reveals space-keys input', () => {
    render(
      <MemoryRouter>
        <ConfluenceConfigure
          {...confluenceProps()}
          authMode="api_token"
          basicAuth={{ siteUrl: 'https://acme.atlassian.net', email: 'a@b.com', apiToken: 'tok' }}
        />
      </MemoryRouter>,
    );
    expect(screen.getByTestId('space-keys-input')).toBeInTheDocument();
  });
});

describe('JiraConfigure — AtlassianConnect integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    noAtlassian();
  });

  it('renders OAuth Connect button by default with no session', () => {
    render(
      <MemoryRouter>
        <JiraConfigure {...jiraProps()} />
      </MemoryRouter>,
    );
    expect(screen.getByTestId('atlassian-connect-button')).toBeInTheDocument();
    expect(screen.queryByTestId('jql-input')).not.toBeInTheDocument();
  });

  it('renders JQL input once OAuth is connected', () => {
    withAtlassian();
    render(
      <MemoryRouter>
        <JiraConfigure {...jiraProps()} />
      </MemoryRouter>,
    );
    expect(screen.getByTestId('jql-input')).toBeInTheDocument();
  });

  it('api_token mode with all fields filled reveals JQL input', () => {
    render(
      <MemoryRouter>
        <JiraConfigure
          {...jiraProps()}
          authMode="api_token"
          basicAuth={{ siteUrl: 'https://acme.atlassian.net', email: 'a@b.com', apiToken: 'tok' }}
        />
      </MemoryRouter>,
    );
    expect(screen.getByTestId('jql-input')).toBeInTheDocument();
  });
});
