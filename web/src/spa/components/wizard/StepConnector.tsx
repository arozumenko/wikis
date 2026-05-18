/**
 * Step 1 of the AddSource wizard (#208): pick a connector.
 *
 * Renders an icon grid of supported source types. Clicking a card both
 * sets ``source_type`` on the wizard state and advances to Step 2 — single
 * action per cartograph's UX. Selected state survives "Back" navigation
 * from later steps so the user doesn't lose their pick mid-flow.
 */

import { Card, CardActionArea, Grid, Stack, Typography } from '@mui/material';
import GitHubIcon from '@mui/icons-material/GitHub';
import ArticleOutlinedIcon from '@mui/icons-material/ArticleOutlined';
import BugReportOutlinedIcon from '@mui/icons-material/BugReportOutlined';
import type { WikiSourceType } from '../../api/wiki';

interface StepConnectorProps {
  selected: WikiSourceType;
  onSelect: (source: WikiSourceType) => void;
}

interface ConnectorOption {
  id: WikiSourceType;
  label: string;
  description: string;
  icon: React.ReactNode;
  testId: string;
}

const CONNECTORS: ConnectorOption[] = [
  {
    id: 'git',
    label: 'Git',
    description: 'GitHub, GitLab, Bitbucket, Azure DevOps, or a local path',
    icon: <GitHubIcon sx={{ fontSize: 40 }} />,
    testId: 'connector-git',
  },
  {
    id: 'confluence',
    label: 'Confluence',
    description: 'Atlassian Cloud spaces',
    icon: <ArticleOutlinedIcon sx={{ fontSize: 40 }} />,
    testId: 'connector-confluence',
  },
  {
    id: 'jira',
    label: 'Jira',
    description: 'Atlassian Cloud issues matching a JQL query',
    icon: <BugReportOutlinedIcon sx={{ fontSize: 40 }} />,
    testId: 'connector-jira',
  },
];

export function StepConnector({ selected, onSelect }: StepConnectorProps) {
  return (
    <Grid container spacing={2} sx={{ mt: 0.5 }}>
      {CONNECTORS.map((c) => {
        const active = c.id === selected;
        return (
          <Grid item xs={12} sm={4} key={c.id}>
            <Card
              variant="outlined"
              sx={{
                borderRadius: 3,
                borderColor: active ? 'primary.main' : 'divider',
                borderWidth: active ? 2 : 1,
                transition: 'border-color 0.15s ease, box-shadow 0.15s ease',
                '&:hover': {
                  borderColor: 'primary.main',
                  boxShadow: '0 0 18px rgba(168, 85, 247, 0.25)',
                },
              }}
            >
              <CardActionArea
                onClick={() => onSelect(c.id)}
                data-testid={c.testId}
                sx={{ p: 2.5, minHeight: 160 }}
              >
                <Stack alignItems="center" spacing={1.5}>
                  {c.icon}
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    {c.label}
                  </Typography>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ textAlign: 'center', minHeight: 32 }}
                  >
                    {c.description}
                  </Typography>
                </Stack>
              </CardActionArea>
            </Card>
          </Grid>
        );
      })}
    </Grid>
  );
}
