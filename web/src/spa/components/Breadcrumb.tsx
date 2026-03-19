import { Breadcrumbs, Link, Typography } from '@mui/material';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import { Link as RouterLink } from 'react-router-dom';

interface BreadcrumbProps {
  repoUrl?: string;
  section?: string;
  pageTitle?: string;
  wikiId?: string;
}

function extractOwnerRepo(url: string): string {
  try {
    const parsed = new URL(url);
    const parts = parsed.pathname.split('/').filter(Boolean);
    if (parts.length >= 2) return `${parts[0]}/${parts[1]}`;
  } catch {
    // Not a valid URL
  }
  return url;
}

export function Breadcrumb({ repoUrl, section, pageTitle, wikiId }: BreadcrumbProps) {
  const segments: { label: string; to?: string }[] = [{ label: 'Wikis', to: '/' }];

  if (repoUrl) {
    segments.push({
      label: extractOwnerRepo(repoUrl),
      to: wikiId ? `/wiki/${wikiId}` : undefined,
    });
  }

  if (section) {
    segments.push({ label: section });
  }

  if (pageTitle) {
    segments.push({ label: pageTitle });
  }

  // Truncate middle segments if >4
  const display =
    segments.length > 4
      ? [...segments.slice(0, 2), { label: '...' }, ...segments.slice(-1)]
      : segments;

  return (
    <Breadcrumbs separator={<NavigateNextIcon fontSize="small" />} sx={{ mb: 2 }}>
      {display.map((seg, i) => {
        const isLast = i === display.length - 1;

        if (isLast || !seg.to) {
          return (
            <Typography
              key={i}
              variant="body2"
              color={isLast ? 'text.primary' : 'text.secondary'}
              fontWeight={isLast ? 600 : 400}
            >
              {seg.label}
            </Typography>
          );
        }

        return (
          <Link
            key={i}
            component={RouterLink}
            to={seg.to}
            underline="hover"
            color="text.secondary"
            variant="body2"
          >
            {seg.label}
          </Link>
        );
      })}
    </Breadcrumbs>
  );
}
