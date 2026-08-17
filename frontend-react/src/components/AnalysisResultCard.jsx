import { useState } from "react";
import * as api from "../api/client";

const SUPPORT_LABEL = {
  DIRECTLY_STATED: "Directly stated",
  STRONGLY_SUPPORTED: "Strongly supported",
  NOT_FOUND: "Not found in evidence"
};

function SupportBadge({ support }) {
  return <span className={`support-badge support-${support?.toLowerCase()}`}>{SUPPORT_LABEL[support] ?? support}</span>;
}

/** Renders the specialized card for one analysis-mode message, based on
 * frontendRole — mirrors the old frontend's renderQuestionSetCard() /
 * renderInterviewEvalCard() / renderPaperEvaluationCard() /
 * renderResearchGapsCard() / renderLiteratureMatrixCard() /
 * renderKnowledgeGraphCard() family exactly, one component per role. */
export function AnalysisResultCard({ message }) {
  const { frontendRole, payload, kicker, text } = message;

  if (frontendRole === "analysis_questions") {
    const questions = payload?.questions ?? [];
    return (
      <div className="analysis-card">
        <div className="analysis-card-title">{payload?.label ?? "Questions"}</div>
        <ol className="question-list">
          {questions.map((q, i) =>
          <li key={i}>
              <div className="question-text">{q.question}</div>
              <div className="question-meta">
                {q.category && <span className="tag">{q.category}</span>}
                {q.difficulty && <span className="tag">{q.difficulty}</span>}
              </div>
              {q.expected_answer_points?.length > 0 &&
            <ul className="expected-points">
                  {q.expected_answer_points.map((p, j) =>
              <li key={j}>{p}</li>
              )}
                </ul>
            }
            </li>
          )}
        </ol>
      </div>);

  }

  if (frontendRole === "interview_question") {
    return (
      <div className="analysis-card">
        <div className="answer-card-head">
          <span className="kicker">{payload?.ended ? "Interview Complete" : "Interview Question"}</span>
          {payload?.category ? <span className="tag">{payload.category}</span> : null}
        </div>
        <div className="bubble-text">{text}</div>
      </div>);

  }

  if (frontendRole === "interview_eval") {
    return (
      <div className="analysis-card">
        <div className="answer-card-head">
          <span className="kicker">Evaluation</span>
          <span className={`tag correctness-${payload?.correctness}`}>{payload?.correctness}</span>
        </div>
        {Array.isArray(payload?.missing_points) && payload.missing_points.length > 0 &&
        <div className="eval-section">
            <strong>Missing points:</strong>
            <ul>
              {payload.missing_points.map((p, i) =>
            <li key={i}>{p}</li>
            )}
            </ul>
          </div>
        }
        {payload?.technical_depth ?
        <div className="eval-section">
            <strong>Technical depth:</strong> {payload.technical_depth}
          </div> :
        null}
        {payload?.suggested_answer ?
        <div className="eval-section">
            <strong>Suggested answer:</strong> {payload.suggested_answer}
          </div> :
        null}
      </div>);

  }

  if (frontendRole === "evaluate_paper") {
    const ev = payload?.evaluation;
    if (!ev) return <div className="bubble-text">{text}</div>;
    const dims = [
    ["problem_clarity", "Problem Clarity"],
    ["novelty", "Novelty"],
    ["methodology", "Methodology"],
    ["experimental_design", "Experimental Design"],
    ["results", "Results"],
    ["limitations", "Limitations"],
    ["reproducibility", "Reproducibility"]];

    return (
      <div className="analysis-card">
        <div className="analysis-card-title">{ev.title || "Paper Evaluation"}</div>
        {dims.map(([key, label]) => {
          const dim = ev[key];
          if (!dim) return null;
          return (
            <div key={key} className="eval-dimension">
              <div className="eval-dimension-head">
                <strong>{label}</strong>
                <SupportBadge support={dim.support} />
              </div>
              {dim.assessment && <div className="eval-dimension-text">{dim.assessment}</div>}
            </div>);

        })}
        {Array.isArray(ev.strengths) && ev.strengths.length > 0 &&
        <div className="eval-section">
            <strong>Strengths:</strong>
            <ul>
              {ev.strengths.map((s, i) =>
            <li key={i}>{s}</li>
            )}
            </ul>
          </div>
        }
        {Array.isArray(ev.weaknesses) && ev.weaknesses.length > 0 &&
        <div className="eval-section">
            <strong>Weaknesses:</strong>
            <ul>
              {ev.weaknesses.map((s, i) =>
            <li key={i}>{s}</li>
            )}
            </ul>
          </div>
        }
        {ev.overall_assessment ?
        <div className="eval-section">
            <strong>Overall:</strong> {ev.overall_assessment}
          </div> :
        null}
      </div>);

  }

  if (frontendRole === "research_gaps") {
    const gaps = payload?.gaps ?? [];
    const titles = payload?.document_titles ?? {};
    return (
      <div className="analysis-card">
        <div className="analysis-card-title">Research Gaps</div>
        {gaps.length === 0 && <div className="bubble-text">No gaps identified from the available evidence.</div>}
        <ul className="gap-list">
          {gaps.map((g, i) =>
          <li key={i}>
              <div className="gap-head">
                <span className={`tag gap-${g.label === "AUTHOR_STATED_GAP" ? "stated" : "inferred"}`}>
                  {g.label === "AUTHOR_STATED_GAP" ? "Author-stated" : "Inferred"}
                </span>
                {g.document_id && <span className="tag">{titles[g.document_id] ?? g.document_id}</span>}
              </div>
              <div>{g.gap}</div>
              {g.evidence && <div className="gap-evidence">{g.evidence}</div>}
            </li>
          )}
        </ul>
      </div>);

  }

  if (frontendRole === "literature_matrix") {
    const rows = payload?.rows ?? [];
    const cols = [
    ["problem", "Problem"],
    ["method", "Method"],
    ["architecture", "Architecture"],
    ["dataset", "Dataset"],
    ["metrics", "Metrics"],
    ["results", "Results"],
    ["limitations", "Limitations"],
    ["research_gap", "Research Gap"]];

    return (
      <div className="analysis-card analysis-card-wide">
        <div className="analysis-card-title">Literature Matrix</div>
        <div className="table-scroll">
          <table className="matrix-table">
            <thead>
              <tr>
                <th>Paper</th>
                {cols.map(([, label]) =>
                <th key={label}>{label}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) =>
              <tr key={i}>
                  <td>{row.title || row.document_id}</td>
                  {cols.map(([key]) =>
                <td key={key}>{row[key]}</td>
                )}
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>);

  }

  if (frontendRole === "knowledge_graph") {
    const nodes = payload?.nodes ?? [];
    const edges = payload?.edges ?? [];
    const titles = payload?.document_titles ?? {};
    return (
      <div className="analysis-card">
        <div className="analysis-card-title">Knowledge Graph</div>
        <div className="kg-nodes">
          {nodes.map((n) =>
          <span key={n.id} className={`tag kg-node-${n.type}`} title={n.document_id ? titles[n.document_id] : undefined}>
              {n.label}
            </span>
          )}
        </div>
        <ul className="kg-edges">
          {edges.map((e, i) => {
            const from = nodes.find((n) => n.id === e.source)?.label ?? e.source;
            const to = nodes.find((n) => n.id === e.target)?.label ?? e.target;
            return (
              <li key={i}>
                <strong>{from}</strong> — {e.relation || "relates to"} → <strong>{to}</strong>
              </li>);

          })}
        </ul>
      </div>);

  }

  if (frontendRole === "report") {
    const report = payload?.report;
    if (!report) return <div className="bubble-text">{text}</div>;
    return (
      <div className="analysis-card analysis-card-wide">
        <div className="analysis-card-title">{report.title || "Comparison Report"}</div>
        <p>{report.overview}</p>
        {Array.isArray(report.papers) &&
        report.papers.map((p, i) =>
        <div key={i} className="report-paper">
              <strong>{p.title || p.document_id}</strong>
              {p.overview ? <p>{p.overview}</p> : null}
            </div>
        )}
        {report.conclusion ?
        <div className="eval-section">
            <strong>Conclusion:</strong> {report.conclusion}
          </div> :
        null}
        {report.report_id ?
        <div className="report-downloads">
            {["markdown", "pdf", "docx"].map((fmt) =>
          <ReportDownloadButton key={fmt} reportId={report.report_id} format={fmt} />
          )}
          </div> :
        null}
      </div>);

  }

  // "assistant_kicker" (Explain Figure, Recommend, Why Design, System
  // Design) and any unrecognized role fall through to a plain bubble with
  // an optional kicker label — this is the deliberate generic fallback,
  // matching the old frontend's default answer-card rendering.
  return (
    <>
      {kicker &&
      <div className="answer-card-head">
          <span className="kicker">{kicker}</span>
        </div>
      }
      <div className="bubble-text">{text}</div>
    </>);

}

/** GET /report/{id}/{format} requires a real Authorization header now (see
 * main.py's report-ownership hardening) — a plain <a href> can't attach
 * custom headers, so this fetches the bytes itself (api.downloadReport)
 * and triggers the save via a throwaway blob URL. Same visible "Download
 * X" tag button as before; only the click handler changed. */
function ReportDownloadButton({ reportId, format }) {
  const [downloading, setDownloading] = useState(false);
  const [failed, setFailed] = useState(false);

  async function handleClick() {
    setDownloading(true);
    setFailed(false);
    try {
      await api.downloadReport(reportId, format);
    } catch {
      setFailed(true);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <button type="button" className="tag report-download-btn" onClick={handleClick} disabled={downloading}>
      {downloading ? "Downloading…" : failed ? "Retry download" : `Download ${format.toUpperCase()}`}
    </button>
  );
}