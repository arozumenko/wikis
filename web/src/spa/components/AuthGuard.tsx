import { useEffect } from 'react';
import { Box, CircularProgress } from '@mui/material';
import { useAuth } from '../hooks/useAuth';

interface AuthGuardProps {
  children: React.ReactNode;
}

export function AuthGuard({ children }: AuthGuardProps) {
  const { loading, authenticated, signIn } = useAuth();

  useEffect(() => {
    if (!loading && !authenticated) {
      signIn();
    }
  }, [loading, authenticated, signIn]);

  if (loading || !authenticated) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }

  return <>{children}</>;
}
