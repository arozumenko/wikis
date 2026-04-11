import { useCallback, useState } from 'react';
import {
  Alert,
  Box,
  Card,
  Chip,
  IconButton,
  Snackbar,
  Tooltip,
  Typography,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import FolderOutlinedIcon from '@mui/icons-material/FolderOutlined';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import PublicOutlinedIcon from '@mui/icons-material/PublicOutlined';
import { useNavigate } from 'react-router-dom';
import { deleteProject, type ProjectResponse } from '../api/project';
import { useAuth } from '../hooks/useAuth';

interface ProjectCardProps {
  project: ProjectResponse;
  onDelete?: () => void;
}

const PROJECT_GRADIENTS = [
  'linear-gradient(135deg, #6366F1, #8B5CF6)',
  'linear-gradient(135deg, #10B981, #059669)',
  'linear-gradient(135deg, #F59E0B, #EF4444)',
  'linear-gradient(135deg, #3B82F6, #06B6D4)',
  'linear-gradient(135deg, #EC4899, #8B5CF6)',
];

function projectGradient(projectId: string): string {
  let hash = 0;
  for (let i = 0; i < projectId.length; i++) {
    hash = (hash * 31 + projectId.charCodeAt(i)) | 0;
  }
  return PROJECT_GRADIENTS[Math.abs(hash) % PROJECT_GRADIENTS.length];
}

export function ProjectCard({ project, onDelete }: ProjectCardProps) {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const isOwner = user?.id === project.owner_id;
  const gradient = projectGradient(project.id);

  const handleDelete = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation();
      if (deleting) return;
      setDeleting(true);
      try {
        await deleteProject(project.id);
        onDelete?.();
      } catch {
        setDeleteError('Failed to delete project. Please try again.');
      } finally {
        setDeleting(false);
      }
    },
    [project.id, onDelete, deleting],
  );

  return (
    <>
    <Box
      sx={{
        position: 'relative',
        borderRadius: 3,
        '&::before': {
          content: '""',
          position: 'absolute',
          inset: 0,
          borderRadius: 'inherit',
          background: gradient,
          opacity: 0,
          filter: 'blur(18px)',
          transition: 'opacity 0.3s ease',
          zIndex: 0,
        },
        '&:hover::before': { opacity: 0.45 },
        '&:hover > .MuiCard-root': { transform: 'translateY(-2px)' },
      }}
    >
      <Card
        elevation={0}
        sx={{
          position: 'relative',
          zIndex: 1,
          height: 140,
          borderRadius: 3,
          bgcolor: 'background.paper',
          border: '1px solid',
          borderColor: 'divider',
          transition: 'transform 0.2s ease',
          cursor: 'pointer',
        }}
        onClick={() => navigate(`/project/${project.id}`)}
      >
        <Box
          sx={{
            height: '100%',
            px: 2.5,
            py: 2,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
          }}
        >
          <Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
              <FolderOutlinedIcon sx={{ fontSize: '1rem', color: 'text.secondary' }} />
              <Typography variant="body1" sx={{ fontWeight: 600 }} noWrap>
                {project.name}
              </Typography>
            </Box>
            {project.description && (
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{
                  display: '-webkit-box',
                  WebkitLineClamp: 2,
                  WebkitBoxOrient: 'vertical',
                  overflow: 'hidden',
                }}
              >
                {project.description}
              </Typography>
            )}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mt: 0.5 }}>
              <Chip
                label={`${project.wiki_count} wiki${project.wiki_count !== 1 ? 's' : ''}`}
                size="small"
                variant="outlined"
                sx={{ height: 18, fontSize: '0.6rem', '& .MuiChip-label': { px: 0.75 } }}
              />
              <Chip
                icon={
                  project.visibility === 'personal' ? (
                    <LockOutlinedIcon sx={{ fontSize: '0.75rem !important' }} />
                  ) : (
                    <PublicOutlinedIcon sx={{ fontSize: '0.75rem !important' }} />
                  )
                }
                label={project.visibility === 'personal' ? 'Personal' : 'Shared'}
                size="small"
                variant="outlined"
                sx={{
                  height: 18,
                  fontSize: '0.6rem',
                  opacity: 0.7,
                  '& .MuiChip-label': { px: 0.75 },
                  '& .MuiChip-icon': { ml: 0.5 },
                }}
              />
            </Box>
          </Box>

          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <Typography variant="caption" color="text.secondary">
              {new Date(project.created_at).toLocaleDateString()}
            </Typography>
            {isOwner && (
              <Box onClick={(e) => e.stopPropagation()}>
                <Tooltip title="Delete project">
                  <span>
                    <IconButton
                      size="small"
                      disabled={deleting}
                      onClick={handleDelete}
                      sx={{
                        color: 'text.secondary',
                        '&:hover': { color: 'error.main' },
                        '&.Mui-disabled': { color: 'text.disabled' },
                      }}
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </span>
                </Tooltip>
              </Box>
            )}
          </Box>
        </Box>
      </Card>
    </Box>
    <Snackbar
      open={deleteError !== null}
      autoHideDuration={4000}
      onClose={() => setDeleteError(null)}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
    >
      <Alert onClose={() => setDeleteError(null)} severity="error" sx={{ width: '100%' }}>
        {deleteError}
      </Alert>
    </Snackbar>
    </>
  );
}
