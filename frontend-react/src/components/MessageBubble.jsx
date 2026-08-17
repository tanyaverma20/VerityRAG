
import { AnalysisResultCard } from "./AnalysisResultCard";

export function MessageBubble({ message }) {
  const { role, text, pending, citations, confidence, error, frontendRole } = message;

  return (
    <div className={`msg-row ${role}`}>
      <div className={`bubble ${error ? "bubble-error" : ""} ${frontendRole ? "bubble-analysis" : ""}`}>
        {pending ?
        <div className="thinking-dots" aria-label="Thinking">
            <span />
            <span />
            <span />
          </div> :
        frontendRole ?
        <AnalysisResultCard message={message} /> :

        <>
            {role === "assistant" && (confidence || message.kicker) &&
          <div className="answer-card-head">
                <span className="kicker">{message.kicker || "Grounded Synthesis"}</span>
                {confidence && <span className={`confidence-pill confidence-${confidence.toLowerCase()}`}>{confidence}</span>}
              </div>
          }
            <div className="bubble-text">{text}</div>
            {citations && citations.length > 0 &&
          <div className="citations">
                {dedupeCitations(citations).map((c, i) =>
            <span key={i} className="citation-pill" title={c.text}>
                    {c.source ?? c.document_id}
                    {c.page_number != null ? ` · p.${c.page_number}` : ""}
                  </span>
            )}
              </div>
          }
          </>
        }
      </div>
    </div>);

}

function dedupeCitations(citations) {
  const seen = new Set();
  const out = [];
  for (const c of citations) {
    const key = `${c.document_id}:${c.page_number ?? ""}`;
    if (!seen.has(key)) {
      seen.add(key);
      out.push(c);
    }
  }
  return out;
}