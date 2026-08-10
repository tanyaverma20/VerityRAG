import { useCallback, useState } from "react";
import * as api from "../api/client";
import type { Citation, ChatMessageRecord } from "../api/types";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  pending?: boolean;
  citations?: Citation[];
  documentsFound?: number;
  confidence?: string;
  error?: boolean;
}

function fromRecord(m: ChatMessageRecord): ChatMessage {
  const meta = (m.metadata ?? {}) as Record<string, unknown>;
  return {
    id: m.message_id,
    role: m.role === "user" ? "user" : "assistant",
    text: m.content,
    citations: (meta.structured_citations as Citation[] | undefined) ?? undefined,
    confidence: meta.confidence as string | undefined,
  };
}

/** Owns one conversation's message list + sending a new question through
 * /query, restoring history from GET /sessions/{id}/messages exactly like
 * the old frontend's ensureMessagesLoaded()/sendQuestion() did. */
export function useChat(sessionId: string | null, workspaceId: string | null) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sending, setSending] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const loadHistory = useCallback(async (id: string) => {
    setLoaded(false);
    const records = await api.getSessionMessages(id);
    setMessages(records.map(fromRecord));
    setLoaded(true);
  }, []);

  const reset = useCallback(() => {
    setMessages([]);
    setLoaded(true);
  }, []);

  const send = useCallback(
    async (question: string, documentIds: string[], sessionIdOverride?: string) => {
      const effectiveSessionId = sessionIdOverride ?? sessionId;
      const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", text: question };
      const pendingMsg: ChatMessage = { id: crypto.randomUUID(), role: "assistant", text: "", pending: true };
      setMessages((prev) => [...prev, userMsg, pendingMsg]);
      setSending(true);
      try {
        const res = await api.query({
          question,
          document_ids: documentIds.length ? documentIds : null,
          workspace_id: workspaceId,
          session_id: effectiveSessionId,
        });
        const docsFound = Array.isArray(res.documents_found) ? res.documents_found.length : res.documents_found;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === pendingMsg.id
              ? {
                  ...m,
                  pending: false,
                  text: res.answer,
                  citations: res.structured_citations,
                  documentsFound: docsFound,
                  confidence: res.confidence,
                }
              : m
          )
        );
      } catch (e) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === pendingMsg.id
              ? { ...m, pending: false, error: true, text: e instanceof Error ? e.message : "Something went wrong." }
              : m
          )
        );
      } finally {
        setSending(false);
      }
    },
    [sessionId, workspaceId]
  );

  return { messages, sending, loaded, loadHistory, reset, send };
}
