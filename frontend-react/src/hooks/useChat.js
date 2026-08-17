import { useCallback, useState } from "react";
import * as api from "../api/client";

function fromRecord(m) {
  const meta = m.metadata ?? {};
  const frontendRole = meta.frontend_role || undefined;
  return {
    id: m.message_id,
    role: m.role === "user" ? "user" : "assistant",
    text: m.content,
    citations: meta.structured_citations ?? undefined,
    confidence: meta.confidence,
    frontendRole,
    payload: frontendRole ? meta : undefined,
    kicker: meta.kicker
  };
}

/** Owns one conversation's message list + sending a new question through
 * /query, plus running any /analyze mode and pushing its result as a
 * message the same way the backend already persists it — restoring from
 * GET /sessions/{id}/messages exactly like the old frontend's
 * ensureMessagesLoaded()/sendQuestion() did. */
export function useChat(sessionId, workspaceId) {
  const [messages, setMessages] = useState([]);
  const [sending, setSending] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const loadHistory = useCallback(async (id) => {
    setLoaded(false);
    const records = await api.getSessionMessages(id);
    setMessages(records.map(fromRecord));
    setLoaded(true);
  }, []);

  const reset = useCallback(() => {
    setMessages([]);
    setLoaded(true);
  }, []);

  const appendMessage = useCallback((m) => {
    setMessages((prev) => [...prev, m]);
  }, []);

  /** Updates an existing message in place by id (e.g. resolving a pending
   * "thinking" bubble into its real result) — never appends a second
   * message, so a caller that shows an optimistic pending bubble and later
   * has the real answer must use this, not appendMessage again, or the
   * pending bubble would be left stuck on screen alongside a duplicate. */
  const updateMessage = useCallback((id, patch) => {
    setMessages((prev) => prev.map((m) => m.id === id ? { ...m, ...patch } : m));
  }, []);

  const send = useCallback(
    async (question, documentIds, sessionIdOverride, researchType) => {
      const effectiveSessionId = sessionIdOverride ?? sessionId;
      const isDeep = researchType === "deep";
      const userMsg = { id: crypto.randomUUID(), role: "user", text: question };
      const pendingMsg = {
        id: crypto.randomUUID(),
        role: "assistant",
        text: "",
        pending: true,
        kicker: isDeep ? "Deep Research" : undefined
      };
      setMessages((prev) => [...prev, userMsg, pendingMsg]);
      setSending(true);
      try {
        const res = await api.query({
          question,
          research_type: isDeep ? "deep" : "simple",
          document_ids: documentIds.length ? documentIds : null,
          workspace_id: workspaceId,
          session_id: effectiveSessionId
        });
        const docsFound = Array.isArray(res.documents_found) ? res.documents_found.length : res.documents_found;
        setMessages((prev) =>
        prev.map((m) =>
        m.id === pendingMsg.id ?
        {
          ...m,
          pending: false,
          text: res.answer,
          citations: res.structured_citations,
          documentsFound: docsFound,
          confidence: res.confidence
        } :
        m
        )
        );
      } catch (e) {
        setMessages((prev) =>
        prev.map((m) =>
        m.id === pendingMsg.id ?
        { ...m, pending: false, error: true, text: e instanceof Error ? e.message : "Something went wrong." } :
        m
        )
        );
      } finally {
        setSending(false);
      }
    },
    [sessionId, workspaceId]
  );

  /** Runs any /analyze mode and returns the raw response — caller decides
   * how to turn it into a message (varies per mode; see AnalysisRunner). */
  const runAnalyze = useCallback(
    async (body, sessionIdOverride) => {
      const effectiveSessionId = sessionIdOverride ?? sessionId;
      return api.analyze({
        ...body,
        workspace_id: workspaceId,
        session_id: effectiveSessionId
      });
    },
    [sessionId, workspaceId]
  );

  return { messages, sending, loaded, loadHistory, reset, send, appendMessage, updateMessage, runAnalyze, setSending };
}