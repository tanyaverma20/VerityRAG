import { useEffect, useRef, useState } from "react";

import { MessageBubble } from "./MessageBubble";
import { ComposerMenu } from "./ComposerMenu";

export function ChatWindow({
  messages,
  documents,
  sending,
  onSend,
  onComposerAction,
  composerPlaceholder,
  composerActiveKeys,
  isDeepResearch,
  onToggleDeepResearch
}) {
  const [question, setQuestion] = useState("");
  const readyDocs = documents.filter((d) => d.ingestion_status === "INDEXED");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  async function handleSubmit(e) {
    e.preventDefault();
    const q = question.trim();
    if (!q || sending) return;
    setQuestion("");
    await onSend(q);
  }

  return (
    <div className="chat-window">
      <div className="chat-messages">
        {messages.length === 0 ?
        <div className="empty-state">
            {readyDocs.length === 0 ?
          <>
                <h2>Start your research workspace</h2>
                <p>Upload a PDF from the sidebar to begin analyzing, comparing, and understanding your papers.</p>
              </> :

          <>
                <h2>Ask anything about your papers</h2>
                <p>Try "What problem is this paper trying to solve?" or tap + for Viva, Compare, Reports, and more.</p>
              </>
          }
          </div> :

        <>
            {messages.map((m) =>
          <MessageBubble key={m.id} message={m} />
          )}
            <div ref={bottomRef} />
          </>
        }
      </div>

      {isDeepResearch &&
        <div className="deep-research-banner">
          🧪 Deep Research is on — your next question runs adaptive multi-step retrieval.
          <button type="button" className="dr-off-btn" onClick={onToggleDeepResearch}>Turn off</button>
        </div>
      }
      <form className="composer" onSubmit={handleSubmit}>
        {onComposerAction && <ComposerMenu onSelect={onComposerAction} activeKeys={composerActiveKeys} />}
        <textarea
          value={question}
          placeholder={
            composerPlaceholder ??
            (readyDocs.length === 0
              ? "Upload a paper to get started…"
              : isDeepResearch
              ? "Ask your Deep Research question…"
              : "Ask a question about your papers…")
          }
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
          rows={1} />

        <button type="submit" disabled={sending || !question.trim()}>
          Send
        </button>
      </form>
    </div>);

}