import {
  Card,
  CardActionArea,
  CardContent,
  IconButton,
  Typography,
  Box,
  Chip,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import type { components } from '../api/types.generated';

type WikiSummary = components['schemas']['WikiSummary'];

interface WikiCardProps {
  wiki: WikiSummary;
  onView: (wikiId: string) => void;
  onDelete: (wikiId: string) => void;
}

export function WikiCard({ wiki, onView, onDelete }: WikiCardProps) {
  return (
    <Card
      variant="outlined"
      sx={{
        borderRadius: 3,
        bgcolor: 'rgba(255, 255, 255, 0.03)',
        backdropFilter: 'blur(10px)',
        transition: 'transform 0.2s ease, box-shadow 0.2s ease',
        '&:hover': {
          transform: 'translateY(-2px)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.12)',
        },
      }}
    >
      <CardActionArea onClick={() => onView(wiki.wiki_id)}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="flex-start">
            <Typography variant="h6" component="div" noWrap sx={{ flex: 1 }}>
              {wiki.title}
            </Typography>
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(wiki.wiki_id);
              }}
              sx={{ ml: 1, flexShrink: 0 }}
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </Box>
          <Typography variant="body2" color="text.secondary" noWrap sx={{ mt: 0.5 }}>
            {wiki.repo_url}
          </Typography>
          <Box display="flex" gap={1} mt={1.5} flexWrap="wrap" alignItems="center">
            <Chip label={wiki.branch} size="small" variant="outlined" />
            {wiki.page_count > 0 ? (
              <Chip label={`${wiki.page_count} pages`} size="small" />
            ) : (
              <Chip label="Generating..." size="small" color="info" variant="outlined" />
            )}
            <Typography variant="caption" color="text.secondary" sx={{ alignSelf: 'center' }}>
              {new Date(wiki.created_at).toLocaleDateString()}
            </Typography>
          </Box>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}
