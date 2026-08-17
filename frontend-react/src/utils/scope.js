// Deterministic (LLM-free) document-scope resolution — a direct port of the
// original vanilla frontend's resolveQueryScope()/detectDocumentScopeFromText()/
// _asksAboutAllDocuments() (frontend/index.html). Behavior preserved exactly:
// an explicit override wins, then a document named in the question text,
// then an explicit "all documents" phrase, then whatever documents are
// click-selected for the current conversation, and only then every
// currently-indexed document as the default.

function isNameLikeToken(token) {
  return !!token && token.length > 3 && /[a-z]/.test(token);
}

/** Documents whose filename (or its first token) is literally mentioned in
 * the question text — e.g. "What does attention.pdf say about..." scopes to
 * just that paper even with several others uploaded. */
export function detectDocumentScopeFromText(question, readyDocuments) {
  const q = (question || "").toLowerCase();
  const matches = readyDocuments.filter((d) => {
    const base = (d.filename || "").replace(/\.pdf$/i, "").toLowerCase();
    if (isNameLikeToken(base) && q.includes(base)) return true;
    const firstToken = base.split(/[^a-z0-9]+/)[0];
    return isNameLikeToken(firstToken) && q.includes(firstToken);
  });
  return matches.map((d) => d.document_id);
}

/** "Summarize all my papers", "compare every document" — an explicit intent
 * to span everything, overriding any per-conversation document selection. */
export function asksAboutAllDocuments(question) {
  const q = (question || "").toLowerCase();
  const mentionsAllIntent = /\b(all|every|each)\b/.test(q);
  const mentionsDocNoun = /\b(papers?|documents?|pdfs?|files?)\b/.test(q);
  return mentionsAllIntent && mentionsDocNoun;
}

/**
 * Resolves which document_ids a request should be scoped to.
 * @param {string} question - the question/prompt text (may be "" for
 *   non-text-driven modes like Evaluate Paper/Research Gaps).
 * @param {Array} readyDocuments - documents with ingestion_status INDEXED.
 * @param {string[]} selectedDocIds - the current conversation's explicitly
 *   click-selected document ids (may be empty).
 * @param {string[]|null} forceDocIds - an explicit override (e.g. Project
 *   Interview pinning the documents an in-progress interview started with).
 * @returns {string[]} the resolved document_ids, in the same priority order
 *   as the original frontend.
 */
export function resolveQueryScope(question, readyDocuments, selectedDocIds, forceDocIds) {
  if (forceDocIds && forceDocIds.length) return forceDocIds;

  const named = detectDocumentScopeFromText(question, readyDocuments);
  if (named.length) return named;

  if (asksAboutAllDocuments(question)) {
    return readyDocuments.map((d) => d.document_id);
  }

  if (selectedDocIds && selectedDocIds.length) {
    const readyIds = new Set(readyDocuments.map((d) => d.document_id));
    const stillReady = selectedDocIds.filter((id) => readyIds.has(id));
    if (stillReady.length) return stillReady;
  }

  return readyDocuments.map((d) => d.document_id);
}
