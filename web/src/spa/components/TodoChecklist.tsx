import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import RadioButtonUncheckedIcon from '@mui/icons-material/RadioButtonUnchecked';
import CircularProgress from '@mui/material/CircularProgress';
import type { TodoItem } from '../api/sse';

interface TodoChecklistProps {
  todos: TodoItem[];
}

function StatusIcon({ status }: { status: TodoItem['status'] }) {
  if (status === 'completed') {
    return <CheckCircleIcon sx={{ fontSize: 16, color: 'success.main' }} />;
  }
  if (status === 'in_progress') {
    return <CircularProgress size={14} thickness={5} />;
  }
  return <RadioButtonUncheckedIcon sx={{ fontSize: 16, color: 'text.disabled' }} />;
}

export function TodoChecklist({ todos }: TodoChecklistProps) {
  if (todos.length === 0) return null;

  const completed = todos.filter((t) => t.status === 'completed').length;

  return (
    <Box sx={{ px: 1, pt: 1 }}>
      <Accordion
        defaultExpanded
        disableGutters
        elevation={0}
        sx={{
          bgcolor: 'action.hover',
          borderRadius: 1,
          '&:before': { display: 'none' },
          overflow: 'hidden',
        }}
      >
        <AccordionSummary
          expandIcon={<ExpandMoreIcon sx={{ fontSize: 16 }} />}
          sx={{
            minHeight: 32,
            px: 1.5,
            '& .MuiAccordionSummary-content': {
              my: 0.5,
              display: 'flex',
              alignItems: 'center',
              gap: 1,
            },
          }}
        >
          <Typography
            variant="caption"
            sx={{
              fontWeight: 600,
              color: 'text.secondary',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              fontSize: '0.65rem',
              flex: 1,
            }}
          >
            Plan ({completed}/{todos.length})
          </Typography>
        </AccordionSummary>

        <AccordionDetails sx={{ px: 1.5, pt: 0, pb: 1 }}>
          {todos.map((todo, i) => (
            <Box
              key={i}
              sx={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 1,
                py: 0.5,
                opacity: todo.status === 'completed' ? 0.6 : 1,
              }}
            >
              <Box sx={{ pt: '2px', flexShrink: 0 }}>
                <StatusIcon status={todo.status} />
              </Box>
              <Typography
                variant="caption"
                sx={{
                  fontSize: '0.75rem',
                  lineHeight: 1.4,
                  textDecoration: todo.status === 'completed' ? 'line-through' : 'none',
                }}
              >
                {todo.content ?? todo.task}
              </Typography>
            </Box>
          ))}
        </AccordionDetails>
      </Accordion>
    </Box>
  );
}
