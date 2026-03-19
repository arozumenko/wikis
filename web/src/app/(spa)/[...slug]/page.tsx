'use client';
import dynamic from 'next/dynamic';

const App = dynamic(() => import('@/spa/App').then((m) => ({ default: m.App })), { ssr: false });

export default function SPACatchAll() {
  return <App />;
}
