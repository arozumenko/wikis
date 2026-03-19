import { useState } from 'react';
import { Accordion, AccordionDetails, AccordionSummary, Typography } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

interface ThinkingStepsProps {
  steps: string[];
}

export function ThinkingSteps({ steps }: ThinkingStepsProps) {
  const [expanded, setExpanded] = useState(false);

  if (steps.length === 0) return null;

  return (
    <Accordion
      expanded={expanded}
      onChange={() => setExpanded(!expanded)}
      variant="outlined"
      sx={{ my: 1, '&:before': { display: 'none' } }}
    >
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography variant="body2" color="text.secondary">
          {steps.length} thinking step{steps.length !== 1 ? 's' : ''}
        </Typography>
      </AccordionSummary>
      <AccordionDetails>
        {steps.map((step, i) => (
          <Typography
            key={i}
            variant="body2"
            sx={{ mb: 1, pl: 2, borderLeft: '2px solid', borderColor: 'divider' }}
          >
            {step}
          </Typography>
        ))}
      </AccordionDetails>
    </Accordion>
  );
}
