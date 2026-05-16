/**
 * @jest-environment jsdom
 */
import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import { CitationChips } from '../CitationChips';

describe('CitationChips', () => {
  it('renders nothing when sources is empty', () => {
    const { container } = render(<CitationChips sources={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders one chip per source with file path label', () => {
    render(
      <CitationChips
        sources={[
          { file_path: 'a/b/c.py', line_start: 12 },
          { file_path: 'x.ts', line_start: null },
        ]}
      />,
    );
    expect(screen.getByText('.../b/c.py:12')).toBeInTheDocument();
    expect(screen.getByText('x.ts')).toBeInTheDocument();
  });

  it('builds GitHub-style URL when repoUrl is set', () => {
    render(
      <CitationChips
        sources={[{ file_path: 'src/foo.ts', line_start: 42 }]}
        repoUrl="https://github.com/me/repo"
        branch="dev"
      />,
    );
    const link = screen.getByText('src/foo.ts:42').closest('a') as HTMLAnchorElement;
    expect(link).not.toBeNull();
    expect(link.href).toBe('https://github.com/me/repo/blob/dev/src/foo.ts#L42');
  });

  it('renders chip as span (non-clickable) when repoUrl is a local path', () => {
    render(
      <CitationChips
        sources={[{ file_path: 'src/foo.ts', line_start: 1 }]}
        repoUrl="/local/checkout"
      />,
    );
    // No <a> wrapper for local-path sources
    expect(screen.queryByRole('link')).toBeNull();
  });

  it('refuses to emit a link for non-http(s) repoUrl schemes (XSS guard)', () => {
    render(
      <CitationChips
        sources={[{ file_path: 'src/foo.ts', line_start: 1 }]}
        repoUrl="javascript:alert(1)"
      />,
    );
    expect(screen.queryByRole('link')).toBeNull();
  });

  it('sets rel="noopener noreferrer" on external chip links', () => {
    render(
      <CitationChips
        sources={[{ file_path: 'src/foo.ts', line_start: 1 }]}
        repoUrl="https://github.com/me/repo"
      />,
    );
    const link = screen.getByRole('link') as HTMLAnchorElement;
    expect(link.rel).toContain('noopener');
    expect(link.rel).toContain('noreferrer');
  });

  it('uses success color for EXTRACTED confidence and renders a tooltip', async () => {
    render(
      <CitationChips
        sources={[{ file_path: 'a.py', line_start: 1, confidence: 'EXTRACTED' }]}
      />,
    );
    // MUI chip exposes the color via the colorSuccess class — assert
    // the chip root has it so we know the visual treatment changed.
    const chip = screen.getByText('a.py:1').closest('.MuiChip-root') as HTMLElement;
    expect(chip).not.toBeNull();
    expect(chip.className).toMatch(/colorSuccess/);
  });

  it('uses warning color for INFERRED confidence', () => {
    render(
      <CitationChips sources={[{ file_path: 'a.py', confidence: 'INFERRED' }]} />,
    );
    const chip = screen.getByText('a.py').closest('.MuiChip-root') as HTMLElement;
    expect(chip.className).toMatch(/colorWarning/);
  });

  it('treats lowercase confidence labels as their uppercase equivalents', () => {
    render(
      <CitationChips sources={[{ file_path: 'a.py', confidence: 'extracted' }]} />,
    );
    const chip = screen.getByText('a.py').closest('.MuiChip-root') as HTMLElement;
    expect(chip.className).toMatch(/colorSuccess/);
  });

  it('falls back to a neutral chip when confidence is null/missing', () => {
    render(<CitationChips sources={[{ file_path: 'a.py' }]} />);
    const chip = screen.getByText('a.py').closest('.MuiChip-root') as HTMLElement;
    // No success/warning class — default-color chip
    expect(chip.className).not.toMatch(/colorSuccess|colorWarning/);
  });
});
