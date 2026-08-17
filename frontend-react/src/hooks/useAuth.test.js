import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import * as api from "../api/client";
import { useAuth } from "./useAuth";

beforeEach(() => {
  localStorage.clear();
});
afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

describe("useAuth", () => {
  it("starts logged out (not loading forever) when no token is stored", async () => {
    const { result } = renderHook(() => useAuth());
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
  });

  it("validates an existing stored token against GET /auth/me on mount", async () => {
    localStorage.setItem("verityrag_auth_token", "existing-token");
    vi.spyOn(api, "getMe").mockResolvedValue({ user_id: "u1", email: "a@example.com" });

    const { result } = renderHook(() => useAuth());
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user.email).toBe("a@example.com");
  });

  it("treats an invalid/expired stored token as logged out, not an error state", async () => {
    localStorage.setItem("verityrag_auth_token", "stale-token");
    vi.spyOn(api, "getMe").mockRejectedValue(new api.ApiError("Invalid or expired session.", 401));

    const { result } = renderHook(() => useAuth());
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
  });

  it("login() sets the authenticated user on success", async () => {
    vi.spyOn(api, "login").mockResolvedValue({ user_id: "u1", email: "a@example.com" });
    const { result } = renderHook(() => useAuth());
    await waitFor(() => expect(result.current.loading).toBe(false));

    await act(async () => {
      await result.current.login("a@example.com", "correctpassword1");
    });
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user.email).toBe("a@example.com");
    expect(result.current.error).toBeNull();
  });

  it("login() surfaces a real error message and stays logged out on failure", async () => {
    vi.spyOn(api, "login").mockRejectedValue(new api.ApiError("Incorrect email or password.", 401));
    const { result } = renderHook(() => useAuth());
    await waitFor(() => expect(result.current.loading).toBe(false));

    await act(async () => {
      await expect(result.current.login("a@example.com", "wrong")).rejects.toThrow();
    });
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.error).toBe("Incorrect email or password.");
  });

  it("register() sets the authenticated user on success", async () => {
    vi.spyOn(api, "register").mockResolvedValue({ user_id: "u2", email: "b@example.com" });
    const { result } = renderHook(() => useAuth());
    await waitFor(() => expect(result.current.loading).toBe(false));

    await act(async () => {
      await result.current.register("b@example.com", "correctpassword1");
    });
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user.email).toBe("b@example.com");
  });

  it("logout() clears the authenticated user even if the network call fails", async () => {
    vi.spyOn(api, "login").mockResolvedValue({ user_id: "u1", email: "a@example.com" });
    const { result } = renderHook(() => useAuth());
    await waitFor(() => expect(result.current.loading).toBe(false));
    await act(async () => {
      await result.current.login("a@example.com", "correctpassword1");
    });
    expect(result.current.isAuthenticated).toBe(true);

    vi.spyOn(api, "logout").mockRejectedValue(new Error("network down"));
    await act(async () => {
      await result.current.logout(); // must resolve, not reject, even on a network failure
    });
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
  });

  it("a global 401 from ANY api call logs the user out immediately, mid-session, without a manual logout()", async () => {
    // useAuth() subscribes to api.onUnauthorized() on mount — capture the
    // real callback it registers there, so this test can fire the exact
    // same real pub-sub api.request() itself invokes on a genuine 401,
    // rather than asserting on a fake test-only hook.
    let firedListener;
    const originalOnUnauthorized = api.onUnauthorized;
    vi.spyOn(api, "onUnauthorized").mockImplementation((cb) => {
      firedListener = cb;
      return originalOnUnauthorized(cb);
    });

    vi.spyOn(api, "login").mockResolvedValue({ user_id: "u1", email: "a@example.com" });
    const { result } = renderHook(() => useAuth());
    await waitFor(() => expect(result.current.loading).toBe(false));
    await act(async () => {
      await result.current.login("a@example.com", "correctpassword1");
    });
    expect(result.current.isAuthenticated).toBe(true);

    act(() => {
      firedListener();
    });
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
  });
});
