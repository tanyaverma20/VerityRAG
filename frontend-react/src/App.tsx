import { useEffect, useRef, useState } from "react";
import * as api from "./api/client";
import { useWorkspace } from "./hooks/useWorkspace";
import { useChat } from "./hooks/useChat";
import { Sidebar } from "./components/Sidebar";
import { ChatWindow } from "./components/ChatWindow";
import "./App.css";

export default function App() {
  const ws = useWorkspace();
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const chat = useChat(activeSessionId, ws.activeId);
  // A session created mid-send (see handleSend) must NOT trigger the
  // load-history effect below — the effect exists for the user explicitly
  // navigating to an existing conversation; loading history right as a
  // brand-new session's first query is still in flight would fetch an
  // empty message list from the backend (nothing's persisted until the
  // query resolves) and wipe the optimistic user/pending bubbles already
  // showing.
  const skipNextHistoryLoad = useRef(false);

  useEffect(() => {
    if (skipNextHistoryLoad.current) {
      skipNextHistoryLoad.current = false;
      return;
    }
    if (activeSessionId) {
      void chat.loadHistory(activeSessionId);
    } else {
      chat.reset();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSessionId]);

  async function handleUpload(file: File) {
    if (!ws.activeId) return;
    await api.uploadDocument(file, ws.activeId);
    await ws.refreshDocuments();
  }

  async function handleDeleteDocument(documentId: string) {
    await api.deleteDocument(documentId, ws.activeId ?? undefined);
    await ws.refreshDocuments();
  }

  function handleNewChat() {
    setActiveSessionId(null);
  }

  function handleSelectSession(id: string) {
    setActiveSessionId(id);
  }

  async function handleDeleteSession(id: string) {
    await api.deleteSession(id);
    if (id === activeSessionId) setActiveSessionId(null);
    await ws.refreshSessions();
  }

  async function handleSend(question: string, documentIds: string[]) {
    // A brand-new chat has no session yet — create one first (mirrors the
    // old frontend's createConversation()-on-first-message behavior), then
    // send through it directly rather than waiting for React state to
    // settle, so the very first message in a new chat isn't lost.
    let sessionId = activeSessionId;
    if (!sessionId && ws.activeId) {
      const created = await api.createSession({ workspaceId: ws.activeId, title: question.slice(0, 60) });
      sessionId = created.session_id;
      skipNextHistoryLoad.current = true;
      setActiveSessionId(sessionId);
      void ws.refreshSessions();
    }
    await chat.send(question, documentIds, sessionId ?? undefined);
  }

  if (ws.loading) {
    return (
      <div className="app-loading">
        <span>Loading VerityRAG…</span>
      </div>
    );
  }

  if (ws.error) {
    return (
      <div className="app-loading">
        <span>Couldn't reach the VerityRAG backend: {ws.error}</span>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <Sidebar
        workspaces={ws.workspaces}
        activeWorkspace={ws.activeWorkspace}
        documents={ws.documents}
        sessions={ws.sessions}
        activeSessionId={activeSessionId}
        onSwitchWorkspace={(id) => {
          setActiveSessionId(null);
          void ws.switchWorkspace(id);
        }}
        onCreateWorkspace={ws.createWorkspace}
        onUpload={handleUpload}
        onDeleteDocument={handleDeleteDocument}
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
      />
      <main className="main-area">
        <ChatWindow messages={chat.messages} documents={ws.documents} sending={chat.sending} onSend={handleSend} />
      </main>
    </div>
  );
}
