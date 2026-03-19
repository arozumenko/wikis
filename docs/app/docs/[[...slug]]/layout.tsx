import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { source } from '@/lib/source';
import type { ReactNode } from 'react';

function WikisLogo() {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, fontSize: '1.1rem' }}>
      <span style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '1.75rem',
        height: '1.75rem',
        borderRadius: '0.35rem',
        background: '#E8622A',
        color: '#fff',
        fontSize: '0.9rem',
      }}>W</span>
      Wikis
    </span>
  );
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <DocsLayout
      tree={source.pageTree}
      nav={{
        title: <WikisLogo />,
        transparentMode: 'none',
      }}
    >
      {children}
    </DocsLayout>
  );
}
