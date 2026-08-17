import { useCallback, useEffect, useState } from "react";
import * as api from "../api/client";

/**
 * Owns the authenticated-user state for the whole app. On mount, if a
 * token is already stored (a page refresh, not a fresh login), it's
 * validated against GET /auth/me — a stale/expired/revoked token is
 * cleared and the user is treated as logged out, never assumed valid.
 * Also listens for a real 401 from ANY other API call (api.onUnauthorized)
 * so an expired session mid-use logs the user out immediately instead of
 * silently failing every subsequent request.
 */
export function useAuth() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!api.getToken()) {
        setLoading(false);
        return;
      }
      try {
        const me = await api.getMe();
        if (!cancelled) setUser(me);
      } catch {
        // Invalid/expired token — api.request() already cleared it.
        if (!cancelled) setUser(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => api.onUnauthorized(() => setUser(null)), []);

  const register = useCallback(async (email, password) => {
    setError(null);
    try {
      const me = await api.register(email, password);
      setUser(me);
      return me;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Registration failed");
      throw e;
    }
  }, []);

  const login = useCallback(async (email, password) => {
    setError(null);
    try {
      const me = await api.login(email, password);
      setUser(me);
      return me;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Login failed");
      throw e;
    }
  }, []);

  const logout = useCallback(async () => {
    // The user's intent to log out on THIS device always succeeds locally,
    // even if the network call to invalidate the session server-side
    // fails (offline, server down) — api.logout() already clears the
    // stored token in that case; this just never lets that failure
    // surface as an unhandled rejection from a plain onClick={onLogout}.
    try {
      await api.logout();
    } catch {
      // already logged; local state still clears below
    } finally {
      setUser(null);
    }
  }, []);

  return { user, loading, error, register, login, logout, isAuthenticated: !!user };
}
