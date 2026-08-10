import type { ChatMessage } from "../hooks/useChat";

export function MessageBubble({ message }: { message: ChatMessage }) {
  const { role, text, pending, citations, confidence, error } = message;

  return (
    <div className={`msg-row ${role}`}>
      <div className={`bubble ${error ? "bubble-error" : ""}`}>
        {pending ? (
          <div className="thinking-dots" aria-label="Thinking">
            <span />
            <span />
            <span />
          </div>
        ) : (
          <>
            {role === "assistant" && confidence && (
              <div className="answer-card-head">
                <span className="kicker">Grounded Synthesis</span>
                <span className={`confidence-pill confidence-${confidence.toLowerCase()}`}>{confidence}</span>
              </div>
            )}
            <div className="bubble-text">{text}</div>
            {citations && citations.length > 0 && (
              <div className="citations">
                {dedupeCitations(citations).map((c, i) => (
                  <span key={i} className="citation-pill" title={c.text}>
                    {c.source ?? c.document_id}
                    {c.page_number != null ? ` · p.${c.page_number}` : ""}
                  </span>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function dedupeCitations(citations: NonNullable<ChatMessage["citations"]>) {
  const seen = new Set<string>();
  const out: typeof citations = [];
  for (const c of citations) {
    const key = `${c.document_id}:${c.page_number ?? ""}`;
    if (!seen.has(key)) {
      seen.add(key);
      out.push(c);
    }
  }
  return out;
}
