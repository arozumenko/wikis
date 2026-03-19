'use client';

import { useEffect } from 'react';

export default function LogoutPage() {
  useEffect(() => {
    fetch('/api/auth/sign-out', {
      method: 'POST',
      credentials: 'include',
    }).finally(() => {
      window.location.href = '/login';
    });
  }, []);

  return (
    <main
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        fontFamily: 'system-ui, sans-serif',
      }}
    >
      <p>Signing out...</p>
    </main>
  );
}
