import { createContext, useContext, useState, useCallback } from 'react';
import type { ProjectResponse } from '../api/project';
import type { components } from '../api/types.generated';

type WikiSummary = components['schemas']['WikiSummary'];

export interface ProjectContextValue {
  projectId: string | null;
  project: ProjectResponse | null;
  wikis: WikiSummary[];
  setProject: (p: ProjectResponse | null) => void;
  setWikis: (wikis: WikiSummary[]) => void;
}

const defaultValue: ProjectContextValue = {
  projectId: null,
  project: null,
  wikis: [],
  setProject: () => {},
  setWikis: () => {},
};

export const ProjectContext = createContext<ProjectContextValue>(defaultValue);

export function ProjectProvider({ children }: { children: React.ReactNode }) {
  const [project, setProjectState] = useState<ProjectResponse | null>(null);
  const [wikis, setWikisState] = useState<WikiSummary[]>([]);

  const setProject = useCallback((p: ProjectResponse | null) => setProjectState(p), []);
  const setWikis = useCallback((w: WikiSummary[]) => setWikisState(w), []);

  return (
    <ProjectContext.Provider
      value={{
        projectId: project?.id ?? null,
        project,
        wikis,
        setProject,
        setWikis,
      }}
    >
      {children}
    </ProjectContext.Provider>
  );
}

export function useProject(): ProjectContextValue {
  return useContext(ProjectContext);
}
