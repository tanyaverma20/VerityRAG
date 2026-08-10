import { useCallback, useEffect, useState } from "react";
import * as api from "../api/client";
import type { DocumentRecord, SessionRecord, Workspace } from "../api/types";

const LAST_WORKSPACE_KEY = "verityrag_last_workspace_id";

/**
 * Owns workspace list/switching plus the documents+sessions that belong to
 * the currently active workspace — mirrors the old frontend's
 * refreshWorkspaces()/switchWorkspace()/loadWorkspaceData() trio, but as a
 * single typed hook instead of globals.
 */
export function useWorkspace() {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadWorkspaceData = useCallback(async (workspaceId: string) => {
    const [docs, sess] = await Promise.all([
      api.listDocuments(workspaceId),
      api.listSessions(workspaceId),
    ]);
    setDocuments(docs);
    setSessions(sess);
  }, []);

  const refreshWorkspaces = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const ws = await api.listWorkspaces();
      setWorkspaces(ws);
      const saved = localStorage.getItem(LAST_WORKSPACE_KEY);
      const initial = ws.find((w) => w.workspace_id === saved)?.workspace_id ?? ws[0]?.workspace_id ?? null;
      setActiveId(initial);
      if (initial) await loadWorkspaceData(initial);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load workspaces");
    } finally {
      setLoading(false);
    }
  }, [loadWorkspaceData]);

  const switchWorkspace = useCallback(
    async (id: string) => {
      setActiveId(id);
      localStorage.setItem(LAST_WORKSPACE_KEY, id);
      await loadWorkspaceData(id);
    },
    [loadWorkspaceData]
  );

  const createWorkspace = useCallback(
    async (name: string) => {
      const ws = await api.createWorkspace(name);
      setWorkspaces((prev) => [ws, ...prev]);
      await switchWorkspace(ws.workspace_id);
      return ws;
    },
    [switchWorkspace]
  );

  const refreshDocuments = useCallback(async () => {
    if (!activeId) return;
    setDocuments(await api.listDocuments(activeId));
  }, [activeId]);

  const refreshSessions = useCallback(async () => {
    if (!activeId) return;
    setSessions(await api.listSessions(activeId));
  }, [activeId]);

  useEffect(() => {
    void refreshWorkspaces();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const activeWorkspace = workspaces.find((w) => w.workspace_id === activeId) ?? null;

  return {
    workspaces,
    activeId,
    activeWorkspace,
    documents,
    sessions,
    loading,
    error,
    switchWorkspace,
    createWorkspace,
    refreshDocuments,
    refreshSessions,
    refreshWorkspaces,
  };
}
