import { AppShell } from './AppShell';
import { useRepoContext } from '../context/RepoContext';

interface Props {
  mode: 'light' | 'dark';
  onToggleTheme: () => void;
}

export function AppShellWithRepo({ mode, onToggleTheme }: Props) {
  const { repo } = useRepoContext();
  return (
    <AppShell
      mode={mode}
      onToggleTheme={onToggleTheme}
      repoContext={repo.repoUrl ? repo : undefined}
    />
  );
}
