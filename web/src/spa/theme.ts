import { createTheme } from '@mui/material/styles';

const HEADING_FONT = 'var(--font-playfair), "Playfair Display", Georgia, serif';
const BODY_FONT = 'var(--font-inter), "Inter", "Roboto", "Helvetica", "Arial", sans-serif';
const CODE_FONT = 'var(--font-jetbrains), "JetBrains Mono", "Fira Code", "Courier New", monospace';

export function createAppTheme(mode: 'light' | 'dark') {
  const isDark = mode === 'dark';

  return createTheme({
    palette: {
      mode,
      primary: {
        main: '#FF6B4A',
        light: '#FF8A6E',
        dark: '#E5552E',
        contrastText: '#fff',
      },
      secondary: {
        main: '#6366F1',
        light: '#818CF8',
        dark: '#4F46E5',
      },
      background: isDark
        ? { default: '#0B1120', paper: '#111827' }
        : { default: '#F8FAFC', paper: '#FFFFFF' },
      text: isDark
        ? { primary: '#F1F5F9', secondary: '#94A3B8' }
        : { primary: '#0F172A', secondary: '#64748B' },
      divider: isDark ? '#1E293B' : '#E2E8F0',
    },
    typography: {
      fontFamily: BODY_FONT,
      h1: { fontFamily: HEADING_FONT, fontWeight: 700 },
      h2: { fontFamily: HEADING_FONT, fontWeight: 600 },
      h3: { fontFamily: HEADING_FONT, fontWeight: 600 },
      h4: { fontFamily: HEADING_FONT, fontWeight: 600 },
      h5: { fontWeight: 600 },
      h6: { fontWeight: 600 },
      body1: { lineHeight: 1.7 },
      body2: { lineHeight: 1.6 },
    },
    shape: {
      borderRadius: 8,
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            scrollbarWidth: 'thin',
          },
          'code, pre': {
            fontFamily: CODE_FONT,
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          contained: {
            textTransform: 'none' as const,
            fontWeight: 600,
            boxShadow: 'none',
            '&:hover': {
              boxShadow: '0 0 20px rgba(255, 107, 74, 0.3)',
            },
          },
          outlined: {
            textTransform: 'none' as const,
          },
          text: {
            textTransform: 'none' as const,
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            boxShadow: isDark ? '0 4px 24px rgba(0, 0, 0, 0.4)' : '0 4px 24px rgba(0, 0, 0, 0.06)',
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            boxShadow: isDark ? '0 1px 8px rgba(0, 0, 0, 0.3)' : '0 1px 8px rgba(0, 0, 0, 0.06)',
            backgroundColor: isDark ? '#0B1120' : '#FFFFFF',
          },
        },
      },
    },
  });
}
