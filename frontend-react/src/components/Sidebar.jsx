import { useRef, useState } from "react";

export function Sidebar({
  workspaces,
  activeWorkspace,
  documents,
  sessions,
  activeSessionId,
  selectedDocIds,
  onToggleDocSelection,
  onSwitchWorkspace,
  onCreateWorkspace,
  onUpload,
  onDeleteDocument,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  user,
  onLogout
}) {
  const [wsMenuOpen, setWsMenuOpen] = useState(false);
  const [creatingWs, setCreatingWs] = useState(false);
  const [newWsName, setNewWsName] = useState("");
  const fileInputRef = useRef(null);
  const [uploading, setUploading] = useState(false);

  async function handleFiles(files) {
    if (!files || !files.length) return;
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        if (file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")) {
          await onUpload(file);
        }
      }
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleCreateWorkspace() {
    const name = newWsName.trim();
    if (!name) return;
    await onCreateWorkspace(name);
    setNewWsName("");
    setCreatingWs(false);
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand">VerityRAG</div>
        <div className="ws-switcher">
          <button className="ws-current" onClick={() => setWsMenuOpen((v) => !v)}>
            <span>{activeWorkspace?.name ?? "Select workspace"}</span>
            <span className="chevron">▾</span>
          </button>
          {wsMenuOpen &&
          <div className="ws-menu" role="menu">
              {workspaces.map((w) =>
            <button
              key={w.workspace_id}
              className={`ws-menu-item ${w.workspace_id === activeWorkspace?.workspace_id ? "active" : ""}`}
              onClick={() => {
                onSwitchWorkspace(w.workspace_id);
                setWsMenuOpen(false);
              }}>
              
                  <span>{w.name}</span>
                  <span className="ws-counts">{w.paper_count} papers</span>
                </button>
            )}
              {!creatingWs ?
            <button className="ws-menu-item new" onClick={() => setCreatingWs(true)}>
                  + New workspace
                </button> :

            <div className="ws-new-form">
                  <input
                autoFocus
                value={newWsName}
                placeholder="Workspace name"
                onChange={(e) => setNewWsName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCreateWorkspace()} />
              
                  <button onClick={handleCreateWorkspace}>Create</button>
                </div>
            }
            </div>
          }
        </div>
      </div>

      <button className="new-chat-btn" onClick={onNewChat}>
        + New chat
      </button>

      <div className="sidebar-section">
        <div className="sidebar-section-title">Papers</div>
        <label className="upload-zone" data-uploading={uploading}>
          {uploading ? "Uploading…" : "Upload PDFs"}
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            multiple
            hidden
            onChange={(e) => handleFiles(e.target.files)} />
          
        </label>
        {documents.some((d) => d.ingestion_status === "INDEXED") &&
          <div className="paper-list-hint">
            Click a paper to scope your next question to just that one — click
            again to include it back in "all papers".
          </div>
        }
        <ul className="paper-list">
          {documents.map((doc) => {
            const isReady = doc.ingestion_status === "INDEXED";
            const isSelected = isReady && selectedDocIds?.includes(doc.document_id);
            return (
              <li
                key={doc.document_id}
                className={`paper-item ${isSelected ? "selected" : ""}`}
                data-status={doc.ingestion_status}
                onClick={isReady ? () => onToggleDocSelection(doc.document_id) : undefined}
                style={isReady ? { cursor: "pointer" } : undefined}
              >
                <span className="paper-title" title={doc.filename}>
                  {doc.title || doc.filename}
                  {isSelected && <span className="paper-selected-tag">Selected</span>}
                </span>
                <span className="paper-status">{doc.ingestion_status}</span>
                <button
                  className="paper-remove"
                  aria-label={`Remove ${doc.filename}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteDocument(doc.document_id);
                  }}>
                  ×
                </button>
              </li>
            );
          })}
          {documents.length === 0 && <li className="paper-empty">No papers yet — upload one to get started.</li>}
        </ul>
      </div>

      <div className="sidebar-section sidebar-section-grow">
        <div className="sidebar-section-title">Recent chats</div>
        <ul className="chat-list">
          {sessions.map((s) =>
          <li
            key={s.session_id}
            className={`chat-item ${s.session_id === activeSessionId ? "active" : ""}`}
            onClick={() => onSelectSession(s.session_id)}>
            
              <span className="chat-item-title">{s.title || "Untitled conversation"}</span>
              <button
              className="chat-item-delete"
              aria-label="Delete conversation"
              onClick={(e) => {
                e.stopPropagation();
                onDeleteSession(s.session_id);
              }}>
              
                ×
              </button>
            </li>
          )}
        </ul>
      </div>

      {user &&
      <div className="sidebar-footer">
          <span className="sidebar-user-email" title={user.email}>{user.email}</span>
          <button type="button" className="sidebar-logout-btn" onClick={onLogout}>
            Log out
          </button>
        </div>
      }
    </aside>);

}