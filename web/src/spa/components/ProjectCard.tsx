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
import ShareIcon from '@mui/icons-material/Share';
import { useNavigate } from 'react-router-dom';
import { deleteProject, updateProject, type ProjectResponse } from '../api/project';

interface ProjectCardProps {
  project: ProjectResponse;
  gradient: string;
  onDelete?: () => void;
  onUpdate?: (updated: ProjectResponse) => void;
}

export function ProjectCard({ project, gradient, onDelete, onUpdate }: ProjectCardProps) {
  const navigate = useNavigate();
  const [deleting, setDeleting] = useState(false);
  const [toggling, setToggling] = useState(false);
  const [snack, setSnack] = useState<{ message: string; severity: 'success' | 'error' } | null>(null);

  const handleDelete = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation();
      if (deleting) return;
      setDeleting(true);
      try {
        await deleteProject(project.id);
        onDelete?.();
      } catch {
        setSnack({ message: 'Failed to delete project.', severity: 'error' });
      } finally {
        setDeleting(false);
      }
    },
    [project.id, onDelete, deleting],
  );

  const handleToggleVisibility = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation();
      if (toggling) return;
      setToggling(true);
      const newVisibility = project.visibility === 'personal' ? 'shared' : 'personal';
      try {
        const updated = await updateProject(project.id, { visibility: newVisibility });
        onUpdate?.(updated);
        setSnack({ message: `Project is now ${newVisibility}`, severity: 'success' });
      } catch {
        setSnack({ message: 'Failed to update visibility.', severity: 'error' });
      } finally {
        setToggling(false);
      }
    },
    [project.id, project.visibility, onUpdate, toggling],
  );

  const visibility = project.visibility ?? 'personal';

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
        }}
      >
        <Box
          onClick={() => navigate(`/project/${project.id}`)}
          sx={{
            height: '100%',
            px: 2.5,
            py: 2,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            cursor: 'pointer',
          }}
        >
          <Box sx={{ width: '100%' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
              <FolderOutlinedIcon sx={{ fontSize: '1rem', color: 'text.secondary' }} />
              <Typography variant="body1" sx={{ fontWeight: 600 }} noWrap>
                {project.name}
              </Typography>
            </Box>
            <Typography variant="caption" color="text.secondary">
              {project.wiki_count} wiki{project.wiki_count !== 1 ? 's' : ''}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mt: 0.5 }}>
              <Chip
                icon={
                  visibility === 'personal' ? (
                    <LockOutlinedIcon sx={{ fontSize: '0.75rem !important' }} />
                  ) : (
                    <PublicOutlinedIcon sx={{ fontSize: '0.75rem !important' }} />
                  )
                }
                label={visibility === 'personal' ? 'Personal' : 'Shared'}
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
              width: '100%',
            }}
          >
            <Typography variant="caption" color="text.secondary">
              {new Date(project.created_at).toLocaleDateString()}
            </Typography>
            <Box onClick={(e) => e.stopPropagation()}>
              <Tooltip title={visibility === 'personal' ? 'Share with all users' : 'Make private'}>
                <span>
                  <IconButton
                    size="small"
                    disabled={toggling}
                    onClick={handleToggleVisibility}
                    sx={{
                      color: visibility === 'shared' ? 'primary.main' : 'text.secondary',
                    }}
                  >
                    <ShareIcon fontSize="small" />
                  </IconButton>
                </span>
              </Tooltip>
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
          </Box>
        </Box>
      </Card>
    </Box>
    <Snackbar
      open={snack !== null}
      autoHideDuration={4000}
      onClose={() => setSnack(null)}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
    >
      <Alert onClose={() => setSnack(null)} severity={snack?.severity ?? 'error'} sx={{ width: '100%' }}>
        {snack?.message}
      </Alert>
    </Snackbar>
    </>
  );
}
