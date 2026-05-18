/**
 * @jest-environment jsdom
 *
 * Tests for the inline AtlassianConnect component (#209 round-2 fixes).
 *
 * Covers:
 *  - Renders the Connect button when atlassian is null
 *  - Clicking the button calls startAtlassianOAuth
 *  - A wikis-oauth-success postMessage triggers onConnected
 */
import '@testing-library/jest-dom';
import React from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AtlassianConnect } from '../AtlassianConnect';

// Mock startAtlassianOAuth
jest.mock('../../../../lib/atlassian-oauth', () => ({
  ...jest.requireActual('../../../../lib/atlassian-oauth'),
  startAtlassianOAuth: jest.fn(),
}));

import { startAtlassianOAuth } from '../../../../lib/atlassian-oauth';
const mockStartOAuth = startAtlassianOAuth as jest.MockedFunction<typeof startAtlassianOAuth>;

beforeEach(() => {
  jest.clearAllMocks();
  mockStartOAuth.mockResolvedValue(undefined);
});

describe('AtlassianConnect', () => {
  it('renders the Connect to Atlassian button', () => {
    render(<AtlassianConnect onConnected={jest.fn()} />);
    expect(screen.getByTestId('atlassian-connect-button')).toBeInTheDocument();
    expect(screen.getByText('Connect to Atlassian')).toBeInTheDocument();
  });

  it('clicking the button calls startAtlassianOAuth', async () => {
    const user = userEvent.setup();
    render(<AtlassianConnect onConnected={jest.fn()} />);
    await user.click(screen.getByTestId('atlassian-connect-button'));
    expect(mockStartOAuth).toHaveBeenCalledTimes(1);
  });

  it('shows a spinner while waiting for the popup', async () => {
    const user = userEvent.setup();
    // Never resolves — keeps the component in 'waiting' state
    mockStartOAuth.mockImplementation(() => new Promise(() => {}));
    render(<AtlassianConnect onConnected={jest.fn()} />);
    await user.click(screen.getByTestId('atlassian-connect-button'));
    expect(await screen.findByRole('progressbar')).toBeInTheDocument();
    expect(screen.queryByTestId('atlassian-connect-button')).not.toBeInTheDocument();
  });

  it('postMessage wikis-oauth-success calls onConnected after delay', async () => {
    jest.useFakeTimers();
    const onConnected = jest.fn();
    render(<AtlassianConnect onConnected={onConnected} />);

    // Simulate the popup completing
    act(() => {
      window.dispatchEvent(
        new MessageEvent('message', {
          data: { type: 'wikis-oauth-success', provider: 'atlassian' },
          origin: window.location.origin,
        }),
      );
    });

    // Before the 600 ms delay fires
    expect(onConnected).not.toHaveBeenCalled();

    act(() => {
      jest.advanceTimersByTime(700);
    });

    expect(onConnected).toHaveBeenCalledTimes(1);
    jest.useRealTimers();
  });

  it('ignores postMessages from different origins', () => {
    const onConnected = jest.fn();
    render(<AtlassianConnect onConnected={onConnected} />);

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
    render(<AtlassianConnect onConnected={jest.fn()} />);
    await user.click(screen.getByTestId('atlassian-connect-button'));
    expect(await screen.findByRole('alert')).toHaveTextContent('Popup blocked');
  });
});

// ---------------------------------------------------------------------------
// Integration: ConfluenceConfigure shows AtlassianConnect when not connected
// ---------------------------------------------------------------------------

import { MemoryRouter } from 'react-router-dom';

// Mock useConnections for the configure components
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

describe('ConfluenceConfigure — AtlassianConnect integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    noAtlassian();
  });

  it('renders the Connect button inside atlassian-connect-warning when atlassian is null', () => {
    render(
      <MemoryRouter>
        <ConfluenceConfigure
          data={{ space_keys: [] }}
          onChange={jest.fn()}
          spaceKeysError={null}
        />
      </MemoryRouter>,
    );
    expect(screen.getByTestId('atlassian-connect-warning')).toBeInTheDocument();
    expect(screen.getByTestId('atlassian-connect-button')).toBeInTheDocument();
  });

  it('does NOT render the Connect button when atlassian is connected', () => {
    withAtlassian();
    render(
      <MemoryRouter>
        <ConfluenceConfigure
          data={{ space_keys: [] }}
          onChange={jest.fn()}
          spaceKeysError={null}
        />
      </MemoryRouter>,
    );
    expect(screen.queryByTestId('atlassian-connect-warning')).not.toBeInTheDocument();
    expect(screen.queryByTestId('atlassian-connect-button')).not.toBeInTheDocument();
    expect(screen.getByTestId('space-keys-input')).toBeInTheDocument();
  });

  it('clicking Connect calls startAtlassianOAuth', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <ConfluenceConfigure
          data={{ space_keys: [] }}
          onChange={jest.fn()}
          spaceKeysError={null}
        />
      </MemoryRouter>,
    );
    await user.click(screen.getByTestId('atlassian-connect-button'));
    expect(mockStartOAuth).toHaveBeenCalledTimes(1);
  });
});

describe('JiraConfigure — AtlassianConnect integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    noAtlassian();
  });

  it('renders the Connect button inside atlassian-connect-warning when atlassian is null', () => {
    render(
      <MemoryRouter>
        <JiraConfigure
          data={{ jql: 'ORDER BY created DESC' }}
          onChange={jest.fn()}
          jqlError={null}
        />
      </MemoryRouter>,
    );
    expect(screen.getByTestId('atlassian-connect-warning')).toBeInTheDocument();
    expect(screen.getByTestId('atlassian-connect-button')).toBeInTheDocument();
  });

  it('does NOT render the Connect button when atlassian is connected', () => {
    withAtlassian();
    render(
      <MemoryRouter>
        <JiraConfigure
          data={{ jql: 'ORDER BY created DESC' }}
          onChange={jest.fn()}
          jqlError={null}
        />
      </MemoryRouter>,
    );
    expect(screen.queryByTestId('atlassian-connect-warning')).not.toBeInTheDocument();
    expect(screen.queryByTestId('atlassian-connect-button')).not.toBeInTheDocument();
    expect(screen.getByTestId('jql-input')).toBeInTheDocument();
  });
});
