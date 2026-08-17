import { describe, expect, it } from "vitest";
import { asksAboutAllDocuments, detectDocumentScopeFromText, resolveQueryScope } from "./scope";

function doc(id, filename) {
  return { document_id: id, filename };
}

describe("detectDocumentScopeFromText", () => {
  it("matches a document whose filename is mentioned in the question", () => {
    const docs = [doc("d1", "attention.pdf"), doc("d2", "bert.pdf")];
    expect(detectDocumentScopeFromText("What does attention.pdf say about heads?", docs)).toEqual(["d1"]);
  });

  it("matches on the filename's first token even without the extension", () => {
    const docs = [doc("d1", "attention-is-all-you-need.pdf")];
    expect(detectDocumentScopeFromText("Summarize attention for me", docs)).toEqual(["d1"]);
  });

  it("does not match a short/non-word-like token", () => {
    const docs = [doc("d1", "gpt.pdf")];
    expect(detectDocumentScopeFromText("What is gpt about?", docs)).toEqual([]);
  });

  it("returns nothing when no filename appears in the text", () => {
    const docs = [doc("d1", "attention.pdf"), doc("d2", "bert.pdf")];
    expect(detectDocumentScopeFromText("What is the main contribution?", docs)).toEqual([]);
  });
});

describe("asksAboutAllDocuments", () => {
  it("detects an explicit all-documents intent", () => {
    expect(asksAboutAllDocuments("Summarize all my papers")).toBe(true);
    expect(asksAboutAllDocuments("Compare every document")).toBe(true);
  });

  it("requires both the all-intent word and a document noun", () => {
    expect(asksAboutAllDocuments("Tell me all about it")).toBe(false);
    expect(asksAboutAllDocuments("Summarize this paper")).toBe(false);
  });
});

describe("resolveQueryScope", () => {
  const readyDocs = [doc("d1", "attention.pdf"), doc("d2", "bert.pdf"), doc("d3", "gpt3.pdf")];

  it("an explicit forceDocIds override always wins", () => {
    expect(resolveQueryScope("anything", readyDocs, ["d2"], ["d3"])).toEqual(["d3"]);
  });

  it("a document named in the text wins over the current selection", () => {
    expect(resolveQueryScope("What does bert.pdf say?", readyDocs, ["d1"], null)).toEqual(["d2"]);
  });

  it("an explicit all-documents phrase spans every ready document", () => {
    expect(resolveQueryScope("Summarize all papers", readyDocs, ["d1"], null)).toEqual(["d1", "d2", "d3"]);
  });

  it("falls back to the conversation's click-selected documents", () => {
    expect(resolveQueryScope("What is the contribution?", readyDocs, ["d1", "d3"], null)).toEqual(["d1", "d3"]);
  });

  it("drops a selected id that is no longer ready, keeping the rest", () => {
    const docsMinusD3 = readyDocs.filter((d) => d.document_id !== "d3");
    expect(resolveQueryScope("What is the contribution?", docsMinusD3, ["d1", "d3"], null)).toEqual(["d1"]);
  });

  it("falls back to every ready document when nothing else applies", () => {
    expect(resolveQueryScope("What is the contribution?", readyDocs, [], null)).toEqual(["d1", "d2", "d3"]);
  });

  it("falls back to every ready document when the selection is entirely stale", () => {
    const docsMinusD3 = readyDocs.filter((d) => d.document_id !== "d3");
    expect(resolveQueryScope("What is the contribution?", docsMinusD3, ["d3"], null)).toEqual(["d1", "d2"]);
  });
});
