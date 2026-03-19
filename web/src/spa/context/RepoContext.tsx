import { createContext, useContext, useState, useCallback } from 'react';

interface RepoContextData {
  wikiId?: string;
  repoUrl?: string;
  branch?: string;
  indexedAt?: string;
  commitHash?: string | null;
}

interface RepoContextValue {
  repo: RepoContextData;
  setRepo: (data: RepoContextData) => void;
  clearRepo: () => void;
}

const RepoCtx = createContext<RepoContextValue>({
  repo: {},
  setRepo: () => {},
  clearRepo: () => {},
});

export function RepoContextProvider({ children }: { children: React.ReactNode }) {
  const [repo, setRepoState] = useState<RepoContextData>({});

  const setRepo = useCallback((data: RepoContextData) => setRepoState(data), []);
  const clearRepo = useCallback(() => setRepoState({}), []);

  return <RepoCtx.Provider value={{ repo, setRepo, clearRepo }}>{children}</RepoCtx.Provider>;
}

export const useRepoContext = () => useContext(RepoCtx);
