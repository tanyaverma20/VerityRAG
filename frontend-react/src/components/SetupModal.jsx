import { useState } from "react";
import { PROJECT_INTERVIEW_TOPICS, SYSTEM_DESIGN_QUESTIONS, WHY_DESIGN_QUESTIONS } from "../utils/interviewConstants";

const TITLES = {
  viva: "Viva Setup",
  mock_test: "Mock Test Setup",
  project_interview_start: "Project Interview Setup",
  explain_figure: "Explain Figure",
  why_design: "Why This Design?",
  system_design: "System Design"
};

export function SetupModal({ mode, onCancel, onSubmit }) {
  const [difficulty, setDifficulty] = useState("intermediate");
  const [numQuestions, setNumQuestions] = useState(5);
  const [figureReference, setFigureReference] = useState("");
  const [topics, setTopics] = useState(() => new Set());

  const needsDifficulty = mode === "viva" || mode === "mock_test" || mode === "project_interview_start";
  const needsNumQuestions = mode === "viva" || mode === "mock_test";
  const needsFigureRef = mode === "explain_figure";
  const needsTopics = mode === "project_interview_start";
  // Why This Design?/System Design pick from a fixed, real question list —
  // matching the original frontend exactly (frontend/index.html's
  // WHY_DESIGN_QUESTIONS/SYSTEM_DESIGN_QUESTIONS, clicked directly rather
  // than typed) — never a free-text box that could ask something the
  // backend's fixed architecture-context prompt wasn't built to answer.
  const isDesignQuestionMode = mode === "why_design" || mode === "system_design";
  const designQuestions = mode === "why_design" ? WHY_DESIGN_QUESTIONS : SYSTEM_DESIGN_QUESTIONS;

  function toggleTopic(t) {
    setTopics((prev) => {
      const next = new Set(prev);
      if (next.has(t)) next.delete(t);else next.add(t);
      return next;
    });
  }

  function handleSubmit(e) {
    e.preventDefault();
    onSubmit({ difficulty, numQuestions, figureReference: figureReference.trim(), topics: [...topics] });
  }

  return (
    <div className="modal-backdrop" onClick={onCancel}>
      {isDesignQuestionMode ?
      <div className="modal setup-modal design-question-modal" onClick={(e) => e.stopPropagation()}>
          <h3>{TITLES[mode]}</h3>
          <p className="modal-sub">
            Pick a question — answered from VerityRAG's real architecture, current vs. proposed clearly separated.
          </p>
          <div className="design-question-list">
            {designQuestions.map((q) =>
          <button key={q} type="button" className="design-question-item" onClick={() => onSubmit({ question: q })}>
                {q}
              </button>
          )}
          </div>
          <div className="modal-actions">
            <button type="button" className="btn-secondary" onClick={onCancel}>
              Cancel
            </button>
          </div>
        </div> :

      <form className="modal setup-modal" onClick={(e) => e.stopPropagation()} onSubmit={handleSubmit}>
          <h3>{TITLES[mode] ?? "Setup"}</h3>

          {needsDifficulty &&
        <label className="modal-field">
              Difficulty
              <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
                <option value="basic">Basic</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
              </select>
            </label>
        }

          {needsNumQuestions &&
        <label className="modal-field">
              Number of questions
              <input
            type="number"
            min={1}
            max={15}
            value={numQuestions}
            onChange={(e) => setNumQuestions(Number(e.target.value))} />

            </label>
        }

          {needsFigureRef &&
        <label className="modal-field">
              Figure/table reference (optional)
              <input
            type="text"
            placeholder='e.g. "Figure 2" or "Table 3" — leave blank for the most relevant one'
            value={figureReference}
            onChange={(e) => setFigureReference(e.target.value)} />

            </label>
        }

          {needsTopics &&
        <label className="modal-field">
              Topics (optional — leave empty for a broad interview)
              <div className="topic-pills">
                {PROJECT_INTERVIEW_TOPICS.map((t) =>
            <button
              key={t}
              type="button"
              className={`topic-pill ${topics.has(t) ? "active" : ""}`}
              onClick={() => toggleTopic(t)}>

                    {t}
                  </button>
            )}
              </div>
            </label>
        }

          <div className="modal-actions">
            <button type="button" className="btn-secondary" onClick={onCancel}>
              Cancel
            </button>
            <button type="submit" className="btn-primary">
              Start
            </button>
          </div>
        </form>
      }
    </div>);

}
