import { useState } from "react";

const GROUPS = [
{
  title: "Research",
  actions: [
  { key: "deep_research", label: "Deep Research", description: "Toggle on, then ask your question below — adaptive multi-step retrieval with verification" },
  { key: "compare", label: "Compare Papers", description: "Similarities, differences, contradictions" },
  { key: "recommend", label: "Recommendation", description: "Which approach to choose, and why" },
  { key: "report", label: "Comparative Report", description: "Full report, downloadable as PDF/DOCX/Markdown" }]

},
{
  title: "Learning & Interview",
  actions: [
  { key: "viva", label: "Viva", description: "Grounded question set from your paper" },
  { key: "mock_test", label: "Mock Test", description: "Timed-style question set" },
  { key: "project_interview_start", label: "Project Interview", description: "Turn-by-turn interview loop" },
  { key: "why_design", label: "Why This Design?", description: "Ask about VerityRAG's own architecture" },
  { key: "system_design", label: "System Design", description: "Ask about VerityRAG's system design" }]

},
{
  title: "Analysis",
  actions: [
  { key: "explain_figure", label: "Explain Figure", description: "Vision-grounded figure/table explanation" },
  { key: "evaluate_paper", label: "Evaluate Paper", description: "Structured critique across 7 dimensions" },
  { key: "research_gaps", label: "Find Research Gaps", description: "Author-stated and inferred gaps" },
  { key: "literature_matrix", label: "Literature Matrix", description: "Side-by-side comparison table" },
  { key: "knowledge_graph", label: "Knowledge Graph", description: "Concepts and relationships" }]

},
{
  title: "Evaluation",
  actions: [{ key: "eval_dashboard", label: "Eval Dashboard", description: "Real retrieval/groundedness/runtime metrics" }]
}];

export function ComposerMenu({ onSelect, activeKeys }) {
  const [open, setOpen] = useState(false);
  const active = activeKeys ?? [];

  return (
    <div className="composer-menu-wrap">
      <button
        type="button"
        className={`composer-plus-btn ${active.length ? "has-active" : ""}`}
        onClick={() => setOpen((v) => !v)}
        aria-label="More actions">
        +
      </button>
      {open &&
      <>
          <div className="composer-menu-backdrop" onClick={() => setOpen(false)} />
          <div className="composer-menu" role="menu">
            {GROUPS.map((group) =>
          <div key={group.title} className="composer-menu-group">
                <div className="composer-menu-group-title">{group.title}</div>
                {group.actions.map((action) => {
                  const isActive = active.includes(action.key);
                  return (
                    <button
                      key={action.key}
                      className={`composer-menu-item ${isActive ? "active" : ""}`}
                      onClick={() => {
                        setOpen(false);
                        onSelect(action.key);
                      }}>
                      <span className="composer-menu-item-label">{action.label}</span>
                      {action.description && <span className="composer-menu-item-desc">{action.description}</span>}
                      {isActive && <span className="composer-menu-item-check" aria-label="active">✓</span>}
                    </button>
                  );
                })}
              </div>
          )}
          </div>
        </>
      }
    </div>);

}