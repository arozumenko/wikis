/**
 * Step 1 of the AddSource wizard: pick a connector.
 *
 * Cartograph-style 6-card 3-column grid. Three connectors are live (Git,
 * Confluence, Jira); three are placeholders (Azure DevOps, Local Git,
 * Notion) marked "Soon" — disabled and dimmed so the surface communicates
 * the roadmap without offering a broken choice. Clicking a live card both
 * sets ``source_type`` on the wizard state and advances to Step 2.
 */

import {
  Box,
  Card,
  CardActionArea,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material';
import GitHubIcon from '@mui/icons-material/GitHub';
import ArticleOutlinedIcon from '@mui/icons-material/ArticleOutlined';
import BugReportOutlinedIcon from '@mui/icons-material/BugReportOutlined';
import CloudOutlinedIcon from '@mui/icons-material/CloudOutlined';
import FolderOutlinedIcon from '@mui/icons-material/FolderOutlined';
import StickyNote2OutlinedIcon from '@mui/icons-material/StickyNote2Outlined';
import type { WikiSourceType } from '../../api/wiki';

// ---------------------------------------------------------------------------
// Connector catalogue
// ---------------------------------------------------------------------------

interface ConnectorOption {
  id: WikiSourceType | 'ado' | 'local_git' | 'notion';
  label: string;
  description: string;
  icon: React.ReactNode;
  iconColor: string;
  testId: string;
  disabled: boolean;
}

const CONNECTORS: ConnectorOption[] = [
  {
    id: 'git',
    label: 'Git',
    description: 'GitHub, GitLab, Bitbucket, or a local path',
    icon: <GitHubIcon sx={{ fontSize: 28 }} />,
    iconColor: 'text.primary',
    testId: 'connector-git',
    disabled: false,
  },
  {
    id: 'confluence',
    label: 'Confluence',
    description: 'Atlassian Cloud spaces',
    icon: <ArticleOutlinedIcon sx={{ fontSize: 28 }} />,
    iconColor: '#2684FF',
    testId: 'connector-confluence',
    disabled: false,
  },
  {
    id: 'jira',
    label: 'Jira',
    description: 'Issues matching a JQL query',
    icon: <BugReportOutlinedIcon sx={{ fontSize: 28 }} />,
    iconColor: '#0052CC',
    testId: 'connector-jira',
    disabled: false,
  },
  {
    id: 'ado',
    label: 'Azure DevOps',
    description: 'Repos & boards',
    icon: <CloudOutlinedIcon sx={{ fontSize: 28 }} />,
    iconColor: '#0078D7',
    testId: 'connector-ado',
    disabled: true,
  },
  {
    id: 'local_git',
    label: 'Local Git',
    description: 'Filesystem path',
    icon: <FolderOutlinedIcon sx={{ fontSize: 28 }} />,
    iconColor: '#FB923C',
    testId: 'connector-local-git',
    disabled: true,
  },
  {
    id: 'notion',
    label: 'Notion',
    description: 'Databases & pages',
    icon: <StickyNote2OutlinedIcon sx={{ fontSize: 28 }} />,
    iconColor: 'text.primary',
    testId: 'connector-notion',
    disabled: true,
  },
];

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface StepConnectorProps {
  selected: WikiSourceType;
  onSelect: (source: WikiSourceType) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function StepConnector({ selected, onSelect }: StepConnectorProps) {
  return (
    <Stack spacing={2} sx={{ mt: 0.5 }}>
      <Typography variant="body2" color="text.secondary">
        Choose the type of data source you want to connect.
      </Typography>

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: 'repeat(2, 1fr)', sm: 'repeat(3, 1fr)' },
          gap: 1.25,
        }}
      >
        {CONNECTORS.map((c) => {
          const active = !c.disabled && c.id === selected;
          const card = (
            <Card
              variant="outlined"
              sx={{
                borderRadius: 2,
                borderColor: active ? 'primary.main' : 'divider',
                borderWidth: active ? 2 : 1,
                opacity: c.disabled ? 0.4 : 1,
                position: 'relative',
                transition: 'border-color 0.15s ease, box-shadow 0.15s ease, background-color 0.15s ease',
                '&:hover': c.disabled
                  ? {}
                  : {
                      borderColor: 'primary.main',
                      boxShadow: '0 0 14px rgba(168, 85, 247, 0.22)',
                    },
              }}
            >
              <CardActionArea
                onClick={() => !c.disabled && onSelect(c.id as WikiSourceType)}
                disabled={c.disabled}
                data-testid={c.testId}
                sx={{ p: 1.75, minHeight: 120 }}
              >
                <Stack alignItems="center" spacing={0.75}>
                  <Box sx={{ color: c.iconColor, display: 'flex' }}>{c.icon}</Box>
                  <Typography
                    component="span"
                    sx={{ fontSize: '0.875rem', fontWeight: 600, lineHeight: 1.2 }}
                  >
                    {c.label}
                  </Typography>
                  <Typography
                    component="span"
                    sx={{
                      fontSize: '0.6875rem',
                      color: 'text.secondary',
                      lineHeight: 1.3,
                      textAlign: 'center',
                      minHeight: 24,
                    }}
                  >
                    {c.description}
                  </Typography>
                </Stack>

                {c.disabled && (
                  <Box
                    sx={{
                      position: 'absolute',
                      top: 6,
                      right: 6,
                      fontSize: '0.625rem',
                      fontWeight: 700,
                      letterSpacing: '0.05em',
                      textTransform: 'uppercase',
                      color: 'text.secondary',
                      bgcolor: 'background.paper',
                      border: 1,
                      borderColor: 'divider',
                      borderRadius: 0.75,
                      px: 0.6,
                      py: 0.15,
                    }}
                  >
                    Soon
                  </Box>
                )}
              </CardActionArea>
            </Card>
          );

          return (
            <Box key={c.id}>
              {c.disabled ? (
                <Tooltip arrow placement="top" title="Coming soon">
                  <Box>{card}</Box>
                </Tooltip>
              ) : (
                card
              )}
            </Box>
          );
        })}
      </Box>
    </Stack>
  );
}
