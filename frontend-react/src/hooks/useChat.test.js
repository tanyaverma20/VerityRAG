import { describe, expect, it, vi, afterEach } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import * as api from "../api/client";
import { useChat } from "./useChat";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useChat", () => {
  it("appendMessage adds a message without touching existing ones", () => {
    const { result } = renderHook(() => useChat("s1", "w1"));
    act(() => {
      result.current.appendMessage({ id: "a", role: "user", text: "Hi" });
    });
    expect(result.current.messages).toHaveLength(1);
    act(() => {
      result.current.appendMessage({ id: "b", role: "assistant", text: "", pending: true });
    });
    expect(result.current.messages.map((m) => m.id)).toEqual(["a", "b"]);
  });

  it("updateMessage replaces the pending bubble in place instead of appending a duplicate — the exact bug this function fixes for runCompare/runDeepResearchPrompt", () => {
    const { result } = renderHook(() => useChat("s1", "w1"));
    act(() => {
      result.current.appendMessage({ id: "user-1", role: "user", text: "Compare these papers." });
      result.current.appendMessage({ id: "pending-1", role: "assistant", text: "", pending: true });
    });
    expect(result.current.messages).toHaveLength(2);

    act(() => {
      result.current.updateMessage("pending-1", { pending: false, text: "Both papers use attention.", confidence: "HIGH" });
    });

    // Still exactly 2 messages — the pending bubble was replaced, not duplicated.
    expect(result.current.messages).toHaveLength(2);
    const resolved = result.current.messages.find((m) => m.id === "pending-1");
    expect(resolved).toMatchObject({ pending: false, text: "Both papers use attention.", confidence: "HIGH" });
  });

  it("updateMessage is a safe no-op when the id does not match any existing message", () => {
    const { result } = renderHook(() => useChat("s1", "w1"));
    act(() => {
      result.current.appendMessage({ id: "a", role: "user", text: "Hi" });
    });
    act(() => {
      result.current.updateMessage("nonexistent", { text: "should not appear" });
    });
    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].text).toBe("Hi");
  });

  it("send() resolves the pending bubble in place on success, never leaving two assistant messages", async () => {
    vi.spyOn(api, "query").mockResolvedValue({
      answer: "42",
      sources: [],
      structured_citations: [],
      documents_found: 1,
      confidence: "HIGH"
    });
    const { result } = renderHook(() => useChat("s1", "w1"));
    await act(async () => {
      await result.current.send("What is the answer?", ["d1"]);
    });
    await waitFor(() => expect(result.current.sending).toBe(false));
    expect(result.current.messages).toHaveLength(2); // user + resolved assistant, no stray pending
    expect(result.current.messages[1]).toMatchObject({ pending: false, text: "42", confidence: "HIGH" });
  });

  it("send() resolves the pending bubble into a real, honest error message on failure", async () => {
    vi.spyOn(api, "query").mockRejectedValue(new Error("The AI reasoning service is temporarily rate-limited."));
    const { result } = renderHook(() => useChat("s1", "w1"));
    await act(async () => {
      await result.current.send("What is the answer?", ["d1"]);
    });
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[1]).toMatchObject({ pending: false, error: true, text: "The AI reasoning service is temporarily rate-limited." });
  });
});