import { lazy, Suspense, useMemo } from 'react';
import { CssBaseline, CircularProgress, Box, ThemeProvider } from '@mui/material';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { AppShellWithRepo } from './components/AppShellWithRepo';
import { AuthGuard } from './components/AuthGuard';
import { ErrorBoundary } from './components/ErrorBoundary';
import { RepoContextProvider } from './context/RepoContext';
import { ProjectProvider } from './context/ProjectContext';
import { useThemeMode } from './hooks/useThemeMode';
import { createAppTheme } from './theme';

const DashboardPage = lazy(() =>
  import('./pages/DashboardPage').then((m) => ({ default: m.DashboardPage })),
);
const WikiViewerPage = lazy(() =>
  import('./pages/WikiViewerPage').then((m) => ({ default: m.WikiViewerPage })),
);
const SettingsPage = lazy(() =>
  import('./pages/SettingsPage').then((m) => ({ default: m.SettingsPage })),
);
const ProjectPage = lazy(() =>
  import('./pages/ProjectPage').then((m) => ({ default: m.ProjectPage })),
);

function PageLoader() {
  return (
    <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
      <CircularProgress />
    </Box>
  );
}

export function App() {
  const { mode, toggleMode } = useThemeMode();
  const theme = useMemo(() => createAppTheme(mode), [mode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ErrorBoundary>
        <BrowserRouter>
          <AuthGuard>
            <RepoContextProvider>
              <Suspense fallback={<PageLoader />}>
                <Routes>
                  <Route element={<AppShellWithRepo mode={mode} onToggleTheme={toggleMode} />}>
                    <Route index element={<DashboardPage />} />
                    <Route path="wiki/:wikiId" element={<WikiViewerPage mode={mode} />} />
                    <Route path="settings" element={<SettingsPage />} />
                    <Route
                      path="project/:projectId"
                      element={
                        <ProjectProvider>
                          <ProjectPage />
                        </ProjectProvider>
                      }
                    />
                  </Route>
                </Routes>
              </Suspense>
            </RepoContextProvider>
          </AuthGuard>
        </BrowserRouter>
      </ErrorBoundary>
    </ThemeProvider>
  );
}
