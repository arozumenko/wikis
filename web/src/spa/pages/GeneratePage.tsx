import { useCallback, useState } from 'react';
import { Box } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { GenerateForm } from '../components/GenerateForm';
import { GenerationProgress } from '../components/GenerationProgress';
import { generateWiki } from '../api/wiki';
import type { SSEEventData } from '../api/sse';
import type { components } from '../api/types.generated';

type GenerateWikiRequest = components['schemas']['GenerateWikiRequest'];

type PageState = 'form' | 'generating';

export function GeneratePage() {
  const navigate = useNavigate();
  const [state, setState] = useState<PageState>('form');
  const [events, setEvents] = useState<SSEEventData[]>([]);

  const handleSubmit = useCallback(
    async (request: GenerateWikiRequest) => {
      setState('generating');
      setEvents([]);

      try {
        const response = await generateWiki(request);

        // Redirect to wiki page immediately with generating flag
        navigate(`/wiki/${response.wiki_id}?generating=true&invocation=${response.invocation_id}`);
      } catch (err: unknown) {
        // 409 Conflict — wiki already exists, navigate to it
        if (err && typeof err === 'object' && 'status' in err && (err as { status: number }).status === 409) {
          const body = (err as { body?: { detail?: { wiki_id?: string } } }).body;
          const wikiId = body?.detail?.wiki_id;
          if (wikiId) {
            navigate(`/wiki/${wikiId}`);
            return;
          }
          // Fallback: 409 but couldn't extract wiki_id
          setEvents([{
            type: 'error', event: 'error',
            error: 'A wiki already exists for this repository. Use the dashboard to find and refresh it.',
            recoverable: true,
          }]);
          return;
        }
        setEvents([
          {
            type: 'error',
            event: 'error',
            error: 'Failed to start wiki generation',
            recoverable: true,
          },
        ]);
      }
    },
    [navigate],
  );

  const handleRetry = useCallback(() => {
    setState('form');
    setEvents([]);
  }, []);

  return (
    <Box sx={{ px: { xs: 3, sm: 4, md: 6, lg: 8 }, py: 4 }}>
      {state === 'form' && <GenerateForm onSubmit={handleSubmit} />}
      {state === 'generating' && <GenerationProgress events={events} onRetry={handleRetry} />}
    </Box>
  );
}
