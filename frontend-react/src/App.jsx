import { useEffect, useRef, useState } from "react";
import * as api from "./api/client";
import { useAuth } from "./hooks/useAuth";
import { useWorkspace } from "./hooks/useWorkspace";
import { useChat } from "./hooks/useChat";
import { Sidebar } from "./components/Sidebar";
import { ChatWindow } from "./components/ChatWindow";
import { SetupModal } from "./components/SetupModal";
import { EvalDashboard } from "./components/EvalDashboard";
import { AuthScreen } from "./components/AuthScreen";
import { resolveQueryScope } from "./utils/scope";
import "./App.css";

// Modes that need no extra setup input before running — dispatched
// immediately when picked from the composer menu.
const ZERO_INPUT_MODES = new Set(["recommend", "evaluate_paper", "research_gaps", "literature_matrix", "knowledge_graph"]);
const SETUP_MODES = new Set(["viva", "mock_test", "project_interview_start", "explain_figure", "why_design", "system_design"]);

/**
 * Top-level gate: every workspace/document/session/query/analyze/report
 * endpoint now requires a real authenticated user (see backend/main.py's
 * Depends(auth.get_current_user)) — AuthenticatedApp (the entire previous
 * App component, unchanged below) only ever mounts, and only ever starts
 * fetching workspaces/documents/sessions, once useAuth() confirms a real
 * logged-in user. This is why the auth gate lives in a SEPARATE component
 * rather than an early return inside the old App: React hooks can't be
 * called conditionally, so useWorkspace()'s data-fetching effect must not
 * even exist yet pre-login, not just be told to skip its fetch.
 */
export default function App() {
  const auth = useAuth();

  if (auth.loading) {
    return (
      <div className="app-loading">
        <span>Loading VerityRAG…</span>
      </div>
    );
  }

  if (!auth.isAuthenticated) {
    return <AuthScreen onLogin={auth.login} onRegister={auth.register} />;
  }

  return <AuthenticatedApp user={auth.user} onLogout={auth.logout} />;
}

function AuthenticatedApp({ user, onLogout }) {
  const ws = useWorkspace();
  const [activeSessionId, setActiveSessionId] = useState(null);
  const chat = useChat(activeSessionId, ws.activeId);
  const skipNextHistoryLoad = useRef(false);

  const [setupMode, setSetupMode] = useState(null);
  const [runningMode, setRunningMode] = useState(null);
  const [showEvalDashboard, setShowEvalDashboard] = useState(false);
  const [activeInterview, setActiveInterview] = useState(null);
  const [modeError, setModeError] = useState(null);
  // Which uploaded documents are explicitly click-selected for the current
  // conversation — mirrors the original frontend's per-conversation
  // conv.selectedDocIds (frontend/index.html: toggleDocSelection()). Empty
  // means "no explicit selection", which resolveQueryScope() below then
  // falls back to (a document named in the question text, an explicit "all
  // documents" phrase, or every currently-indexed document).
  const [selectedDocIds, setSelectedDocIds] = useState([]);
  // Deep Research toggle — matches the original frontend's plain checkbox
  // pill exactly (frontend/index.html: toggleDeepResearchPill()): flips a
  // persistent flag, does not itself run anything. The NEXT normal question
  // sent through the composer is what actually runs with
  // research_type="deep" on the SAME /query call normal Q&A already uses —
  // never a separate async task/polling flow, which the original frontend
  // never had.
  const [deepResearchActive, setDeepResearchActive] = useState(false);

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
    setActiveInterview(null);
    setSelectedDocIds([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSessionId]);

  const readyDocuments = ws.documents.filter((d) => d.ingestion_status === "INDEXED");
  const readyDocumentIds = readyDocuments.map((d) => d.document_id);

  function toggleDocSelection(documentId) {
    setSelectedDocIds((prev) =>
      prev.includes(documentId) ? prev.filter((id) => id !== documentId) : [...prev, documentId]
    );
  }

  /** The one scope resolver every request path uses — mirrors the original
   * frontend's resolveQueryScope() exactly (see utils/scope.js). */
  function scopeFor(question, forceDocIds) {
    return resolveQueryScope(question, readyDocuments, selectedDocIds, forceDocIds ?? null);
  }

  async function ensureSession(seedTitle) {
    if (activeSessionId) return activeSessionId;
    if (!ws.activeId) return null;
    const created = await api.createSession({ workspaceId: ws.activeId, title: seedTitle.slice(0, 60) });
    skipNextHistoryLoad.current = true;
    setActiveSessionId(created.session_id);
    void ws.refreshSessions();
    return created.session_id;
  }

  async function handleUpload(file) {
    if (!ws.activeId) return;
    await api.uploadDocument(file, ws.activeId);
    await ws.refreshDocuments();
  }

  async function handleDeleteDocument(documentId) {
    await api.deleteDocument(documentId, ws.activeId ?? undefined);
    await ws.refreshDocuments();
  }

  function handleNewChat() {
    setActiveSessionId(null);
  }

  function handleSelectSession(id) {
    setActiveSessionId(id);
  }

  async function handleDeleteSession(id) {
    await api.deleteSession(id);
    if (id === activeSessionId) setActiveSessionId(null);
    await ws.refreshSessions();
  }

  // -------------------------------------------------------------------
  // Normal Q&A send — also the channel Project Interview answers travel
  // through, since from the user's point of view it's the same composer.
  // -------------------------------------------------------------------
  async function handleSend(question) {
    if (activeInterview) {
      await submitInterviewAnswer(question);
      return;
    }
    if (!readyDocumentIds.length) {
      chat.appendMessage({ id: crypto.randomUUID(), role: "user", text: question });
      chat.appendMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        text: "Upload at least one paper to get started — I can only answer questions about documents in your current workspace."
      });
      return;
    }
    const scopedDocIds = scopeFor(question, null);
    const sessionId = await ensureSession(question);
    await chat.send(question, scopedDocIds, sessionId ?? undefined, deepResearchActive ? "deep" : "simple");
  }

  // -------------------------------------------------------------------
  // Composer menu dispatch
  // -------------------------------------------------------------------
  function handleComposerSelect(key) {
    setModeError(null);
    if (key === "eval_dashboard") {
      setShowEvalDashboard(true);
      return;
    }
    if (key === "report") {
      void runReport();
      return;
    }
    if (key === "compare") {
      void runCompare();
      return;
    }
    if (key === "deep_research") {
      setDeepResearchActive((v) => !v);
      return;
    }
    if (SETUP_MODES.has(key)) {
      setSetupMode(key);
      return;
    }
    if (ZERO_INPUT_MODES.has(key)) {
      void runZeroInputMode(key);
      return;
    }
  }

  async function runZeroInputMode(mode) {
    if (readyDocumentIds.length === 0) {
      setModeError("Upload at least one paper first.");
      return;
    }
    if (mode === "literature_matrix" && readyDocumentIds.length < 2) {
      setModeError("Select at least two papers to build a literature matrix.");
      return;
    }
    // Zero-input modes have no question text to scan for a named document,
    // so this only ever resolves to the current selection or every ready
    // document — matching the original frontend's resolveQueryScope("", ...).
    let scopedDocIds = scopeFor("", null);
    // Literature Matrix genuinely needs >=2 papers of evidence — if the
    // current click-selection narrowed it below 2, fall back to comparing
    // every ready paper rather than failing on a selection that was really
    // meant to scope Q&A, not this mode (same fallback as the original).
    if (mode === "literature_matrix" && scopedDocIds.length < 2) {
      scopedDocIds = readyDocumentIds;
    }
    setRunningMode(mode);
    try {
      const sessionId = await ensureSession(mode);
      const res = await chat.runAnalyze({ mode, document_ids: scopedDocIds }, sessionId ?? undefined);
      pushAnalyzeResult(mode, res);
    } catch (e) {
      setModeError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setRunningMode(null);
    }
  }

  async function handleSetupSubmit(fields) {
    const mode = setupMode;
    setSetupMode(null);
    if (mode === "why_design" || mode === "system_design") {
      if (!fields.question) return;
      setRunningMode(mode);
      try {
        const sessionId = await ensureSession(fields.question);
        const res = await chat.runAnalyze({ mode, question: fields.question }, sessionId ?? undefined);
        pushAnalyzeResult(mode, res, `Q: ${fields.question}`);
      } catch (e) {
        setModeError(e instanceof Error ? e.message : "Something went wrong.");
      } finally {
        setRunningMode(null);
      }
      return;
    }

    if (readyDocumentIds.length === 0) {
      setModeError("Upload at least one paper first.");
      return;
    }

    if (mode === "project_interview_start") {
      const scopedDocIds = scopeFor("", null);
      setRunningMode(mode);
      try {
        const sessionId = await ensureSession("Project Interview");
        const res = await chat.runAnalyze(
          { mode, difficulty: fields.difficulty, topics: fields.topics ?? [], document_ids: scopedDocIds },
          sessionId ?? undefined
        );
        if (res.ok) {
          setActiveInterview({
            question: res.question,
            category: "",
            difficulty: fields.difficulty,
            topics: fields.topics ?? [],
            documentIds: scopedDocIds
          });
          chat.appendMessage({
            id: crypto.randomUUID(),
            role: "assistant",
            text: res.question,
            frontendRole: "interview_question",
            payload: {}
          });
        } else {
          setModeError(res.error ?? "Failed to start the interview.");
        }
      } catch (e) {
        setModeError(e instanceof Error ? e.message : "Something went wrong.");
      } finally {
        setRunningMode(null);
      }
      return;
    }

    // Explain Figure's own reference text ("Figure 2", "the architecture
    // diagram") is scanned for a named document too, exactly like a normal
    // question would be — matching the original frontend's
    // resolveQueryScope(figureRef, null).
    const scopedDocIds = scopeFor(mode === "explain_figure" ? fields.figureReference || "" : "", null);
    setRunningMode(mode);
    try {
      const sessionId = await ensureSession(mode);
      const body = { mode, document_ids: scopedDocIds };
      if (mode === "viva" || mode === "mock_test") {
        body.difficulty = fields.difficulty;
        body.num_questions = fields.numQuestions;
      }
      if (mode === "explain_figure") {
        body.figure_reference = fields.figureReference || null;
      }
      const res = await chat.runAnalyze(body, sessionId ?? undefined);
      pushAnalyzeResult(mode, res);
    } catch (e) {
      setModeError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setRunningMode(null);
    }
  }

  function pushAnalyzeResult(mode, res, userText) {
    if (!res.ok) {
      setModeError(res.error ?? "Generation failed.");
      return;
    }
    const messages = [];
    if (userText) {
      messages.push({ id: crypto.randomUUID(), role: "user", text: userText });
    }
    switch (mode) {
      case "viva":
      case "mock_test":{
          const label = `${mode === "viva" ? "Viva" : "Mock Test"} Questions`;
          messages.push({
            id: crypto.randomUUID(),
            role: "assistant",
            text: label,
            frontendRole: "analysis_questions",
            payload: { label, questions: res.questions, evidence_sufficient: res.evidence_sufficient }
          });
          break;
        }
      case "explain_figure":
      case "recommend":
        messages.push({
          id: crypto.randomUUID(),
          role: "assistant",
          text: res.answer,
          kicker: mode === "explain_figure" ? "Figure Explanation" : "Recommendation"
        });
        break;
      case "why_design":
      case "system_design":
        messages.push({ id: crypto.randomUUID(), role: "assistant", text: res.answer, kicker: mode === "why_design" ? "Why This Design?" : "System Design" });
        break;
      case "evaluate_paper":
        messages.push({
          id: crypto.randomUUID(),
          role: "assistant",
          text: "Paper Evaluation",
          frontendRole: "evaluate_paper",
          payload: { evaluation: res.evaluation }
        });
        break;
      case "research_gaps":
        messages.push({
          id: crypto.randomUUID(),
          role: "assistant",
          text: "Research Gaps",
          frontendRole: "research_gaps",
          payload: { gaps: res.gaps, evidence_sufficient: res.evidence_sufficient, document_titles: res.document_titles }
        });
        break;
      case "literature_matrix":
        messages.push({
          id: crypto.randomUUID(),
          role: "assistant",
          text: "Literature Matrix",
          frontendRole: "literature_matrix",
          payload: { rows: res.rows, evidence_sufficient: res.evidence_sufficient }
        });
        break;
      case "knowledge_graph":
        messages.push({
          id: crypto.randomUUID(),
          role: "assistant",
          text: "Knowledge Graph",
          frontendRole: "knowledge_graph",
          payload: { nodes: res.nodes, edges: res.edges, evidence_sufficient: res.evidence_sufficient, document_titles: res.document_titles }
        });
        break;
      default:
        messages.push({ id: crypto.randomUUID(), role: "assistant", text: JSON.stringify(res) });
    }
    messages.forEach(chat.appendMessage);
  }

  // -------------------------------------------------------------------
  // Project Interview turn loop
  // -------------------------------------------------------------------
  async function submitInterviewAnswer(userAnswerText) {
    if (!activeInterview) return;
    chat.appendMessage({ id: crypto.randomUUID(), role: "user", text: userAnswerText });
    chat.setSending(true);
    try {
      const sessionId = await ensureSession("Project Interview");
      const res = await chat.runAnalyze(
        {
          mode: "project_interview_evaluate",
          document_ids: activeInterview.documentIds,
          question: activeInterview.question,
          user_answer: userAnswerText,
          difficulty: activeInterview.difficulty,
          topics: activeInterview.topics
        },
        sessionId ?? undefined
      );
      if (!res.ok) {
        setModeError(res.error ?? "Evaluation failed.");
        return;
      }
      chat.appendMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        text: res.suggested_answer || res.technical_depth || "Evaluation",
        frontendRole: "interview_eval",
        payload: { correctness: res.correctness, missing_points: res.missing_points, technical_depth: res.technical_depth, suggested_answer: res.suggested_answer }
      });
      if (res.next_question) {
        setActiveInterview({ ...activeInterview, question: res.next_question, category: res.next_question_category });
        chat.appendMessage({
          id: crypto.randomUUID(),
          role: "assistant",
          text: res.next_question,
          frontendRole: "interview_question",
          payload: { category: res.next_question_category }
        });
      } else {
        setActiveInterview(null);
        chat.appendMessage({
          id: crypto.randomUUID(),
          role: "assistant",
          text: "Interview complete — nice work. Start Project Interview again from the + menu for a new one.",
          frontendRole: "interview_question",
          payload: { ended: true }
        });
      }
    } catch (e) {
      setModeError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      chat.setSending(false);
    }
  }

  // -------------------------------------------------------------------
  // Comparative report + Compare — both real backend actions call the SAME
  // /report endpoint (a per-paper + cross-paper structured comparison),
  // exactly like the original frontend (frontend/index.html: generateReport()
  // and runCompare() both POST /report) — "Compare" is not a different,
  // separate query, just the same report scoped by the current document
  // selection with a >=2-papers fallback to every ready paper.
  // -------------------------------------------------------------------
  async function runReportLike(kicker) {
    if (readyDocumentIds.length < 2) {
      setModeError(`Select at least two papers to ${kicker === "Compare" ? "compare" : "generate a comparison report"}.`);
      return;
    }
    let scopedDocIds = scopeFor("", null);
    if (scopedDocIds.length < 2) scopedDocIds = readyDocumentIds;

    setRunningMode(kicker === "Compare" ? "compare" : "report");
    try {
      const sessionId = await ensureSession(kicker === "Compare" ? "Compare papers" : "Comparison Report");
      const res = await api.generateReport({ document_ids: scopedDocIds, workspace_id: ws.activeId, session_id: sessionId });
      if (!res.ok) {
        setModeError(res.error ?? "Report generation failed.");
        return;
      }
      chat.appendMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        text: res.title ?? "Comparison Report",
        frontendRole: "report",
        payload: { report: res }
      });
    } catch (e) {
      setModeError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setRunningMode(null);
    }
  }

  const runReport = () => runReportLike("Report");
  const runCompare = () => runReportLike("Compare");

  if (ws.loading) {
    return (
      <div className="app-loading">
        <span>Loading VerityRAG…</span>
      </div>);

  }

  if (ws.error) {
    return (
      <div className="app-loading">
        <span>Couldn't reach the VerityRAG backend: {ws.error}</span>
      </div>);

  }

  return (
    <div className="app-shell">
      <Sidebar
        workspaces={ws.workspaces}
        activeWorkspace={ws.activeWorkspace}
        documents={ws.documents}
        sessions={ws.sessions}
        activeSessionId={activeSessionId}
        selectedDocIds={selectedDocIds}
        onToggleDocSelection={toggleDocSelection}
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
        user={user}
        onLogout={onLogout} />

      <main className="main-area">
        {modeError &&
        <div className="mode-error-banner" onClick={() => setModeError(null)}>
            {modeError} <span className="dismiss">✕</span>
          </div>
        }
        {activeInterview &&
        <div className="interview-banner">Project Interview in progress — your next message is your answer.</div>
        }
        <ChatWindow
          messages={chat.messages}
          documents={ws.documents}
          sending={chat.sending || runningMode !== null}
          onSend={handleSend}
          onComposerAction={handleComposerSelect}
          composerActiveKeys={deepResearchActive ? ["deep_research"] : []}
          isDeepResearch={deepResearchActive}
          onToggleDeepResearch={() => setDeepResearchActive((v) => !v)}
          composerPlaceholder={activeInterview ? "Type your answer…" : undefined} />

      </main>
      {setupMode && <SetupModal mode={setupMode} onCancel={() => setSetupMode(null)} onSubmit={handleSetupSubmit} />}
      {showEvalDashboard && <EvalDashboard onClose={() => setShowEvalDashboard(false)} />}
    </div>);

}