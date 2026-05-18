/**
 * @jest-environment jsdom
 */
import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AskBar } from '../AskBar';

const STORAGE_KEY = 'wikis:askbar:verifiedOnly';

describe('AskBar — verified-only filter', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('defaults to no filter and submits null minConfidence', async () => {
    const user = userEvent.setup();
    const onSubmit = jest.fn();
    render(<AskBar onSubmit={onSubmit} />);

    const textbox = screen.getByRole('textbox');
    await user.type(textbox, 'hello world');
    await user.keyboard('{Enter}');

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith('hello world', 'fast', null);
  });

  it('toggles verified-only and sends min_confidence=EXTRACTED', async () => {
    const user = userEvent.setup();
    const onSubmit = jest.fn();
    render(<AskBar onSubmit={onSubmit} />);

    const toggle = screen.getByRole('button', { name: /verified-only filter/i });
    await user.click(toggle);

    const textbox = screen.getByRole('textbox');
    await user.type(textbox, 'why is sky blue');
    await user.keyboard('{Enter}');

    expect(onSubmit).toHaveBeenCalledWith('why is sky blue', 'fast', 'EXTRACTED');
  });

  it('persists toggle state across mounts via localStorage', async () => {
    const user = userEvent.setup();
    const onSubmit = jest.fn();

    const { unmount } = render(<AskBar onSubmit={onSubmit} />);
    const toggle = screen.getByRole('button', { name: /verified-only filter/i });
    await user.click(toggle);
    unmount();

    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('1');

    render(<AskBar onSubmit={onSubmit} />);
    const textbox = screen.getByRole('textbox');
    await user.type(textbox, 'restored');
    await user.keyboard('{Enter}');

    expect(onSubmit).toHaveBeenCalledWith('restored', 'fast', 'EXTRACTED');
  });

  it('keeps icon filled and aria-pressed=true when verifiedOnly is on but mode != fast', async () => {
    window.localStorage.setItem(STORAGE_KEY, '1');
    const user = userEvent.setup();
    const onSubmit = jest.fn();
    render(<AskBar onSubmit={onSubmit} />);

    // Switch to Deep mode
    await user.click(screen.getByText('Fast'));
    await user.click(screen.getByText('Deep Research'));

    const toggle = screen.getByRole('button', { name: /verified-only filter/i });
    expect(toggle).toBeDisabled();
    // The preference is preserved, so aria-pressed must reflect it
    expect(toggle).toHaveAttribute('aria-pressed', 'true');
  });

  it('disables toggle and passes null minConfidence when not in fast mode', async () => {
    // Start with the toggle pre-enabled via storage; we should still
    // get a null minConfidence because non-fast modes don't honour
    // the filter yet.
    window.localStorage.setItem(STORAGE_KEY, '1');

    const user = userEvent.setup();
    const onSubmit = jest.fn();
    render(<AskBar onSubmit={onSubmit} />);

    // Open mode menu (the start adornment is clickable to open Menu)
    const modeChip = screen.getByText('Fast');
    await user.click(modeChip);
    await user.click(screen.getByText('Deep Research'));

    const toggle = screen.getByRole('button', { name: /verified-only filter/i });
    expect(toggle).toBeDisabled();

    const textbox = screen.getByRole('textbox');
    await user.type(textbox, 'deep question');
    await user.keyboard('{Enter}');

    expect(onSubmit).toHaveBeenCalledWith('deep question', 'deep', null);
  });
});
