import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../hooks/useChat";
import type { DocumentRecord } from "../api/types";
import { MessageBubble } from "./MessageBubble";

interface ChatWindowProps {
  messages: ChatMessage[];
  documents: DocumentRecord[];
  sending: boolean;
  onSend: (question: string, documentIds: string[]) => Promise<void>;
}

export function ChatWindow({ messages, documents, sending, onSend }: ChatWindowProps) {
  const [question, setQuestion] = useState("");
  const readyDocs = documents.filter((d) => d.ingestion_status === "INDEXED");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q || sending) return;
    setQuestion("");
    await onSend(
      q,
      readyDocs.map((d) => d.document_id)
    );
  }

  return (
    <div className="chat-window">
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="empty-state">
            {readyDocs.length === 0 ? (
              <>
                <h2>Start your research workspace</h2>
                <p>Upload a PDF from the sidebar to begin analyzing, comparing, and understanding your papers.</p>
              </>
            ) : (
              <>
                <h2>Ask anything about your papers</h2>
                <p>Try "What problem is this paper trying to solve?"</p>
              </>
            )}
          </div>
        ) : (
          <>
            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} />
            ))}
            <div ref={bottomRef} />
          </>
        )}
      </div>

      <form className="composer" onSubmit={handleSubmit}>
        <textarea
          value={question}
          placeholder={readyDocs.length === 0 ? "Upload a paper to get started…" : "Ask a question about your papers…"}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
          rows={1}
        />
        <button type="submit" disabled={sending || !question.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
