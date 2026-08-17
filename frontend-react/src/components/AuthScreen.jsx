import { useState } from "react";

/**
 * Gates the whole app until a real, authenticated user exists — rendered
 * instead of the main layout (App.jsx) when useAuth().user is null. Reuses
 * the app's existing modal-field/btn-primary/btn-secondary vocabulary
 * (see SetupModal.jsx) rather than introducing a new visual language.
 */
export function AuthScreen({ onLogin, onRegister }) {
  const [mode, setMode] = useState("login"); // "login" | "register"
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [localError, setLocalError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    setLocalError(null);
    setSubmitting(true);
    try {
      if (mode === "login") {
        await onLogin(email, password);
      } else {
        await onRegister(email, password);
      }
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-screen">
      <form className="modal auth-card" onSubmit={handleSubmit}>
        <div className="brand auth-brand">VerityRAG</div>
        <h3>{mode === "login" ? "Log in" : "Create an account"}</h3>

        <label className="modal-field">
          Email
          <input
            type="email"
            autoComplete="username"
            autoFocus
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </label>

        <label className="modal-field">
          Password
          <input
            type="password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>

        {localError && <div className="auth-error">{localError}</div>}

        <div className="modal-actions auth-actions">
          <button type="submit" className="btn-primary" disabled={submitting}>
            {submitting ? "Please wait…" : mode === "login" ? "Log in" : "Create account"}
          </button>
        </div>

        <button
          type="button"
          className="auth-switch"
          onClick={() => {
            setLocalError(null);
            setMode((m) => (m === "login" ? "register" : "login"));
          }}
        >
          {mode === "login" ? "Need an account? Register" : "Already have an account? Log in"}
        </button>
      </form>
    </div>
  );
}
