import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { AuthScreen } from "./AuthScreen";

describe("AuthScreen", () => {
  it("defaults to the login form", () => {
    render(<AuthScreen onLogin={vi.fn()} onRegister={vi.fn()} />);
    expect(screen.getByRole("heading", { name: "Log in" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Log in" })).toBeInTheDocument();
  });

  it("submits email/password to onLogin", async () => {
    const onLogin = vi.fn().mockResolvedValue();
    render(<AuthScreen onLogin={onLogin} onRegister={vi.fn()} />);

    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "a@example.com" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "correctpassword1" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));

    await waitFor(() => expect(onLogin).toHaveBeenCalledWith("a@example.com", "correctpassword1"));
  });

  it("switches to the register form and submits to onRegister instead", async () => {
    const onRegister = vi.fn().mockResolvedValue();
    render(<AuthScreen onLogin={vi.fn()} onRegister={onRegister} />);

    fireEvent.click(screen.getByText("Need an account? Register"));
    expect(screen.getByRole("heading", { name: "Create an account" })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "new@example.com" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "correctpassword1" } });
    fireEvent.click(screen.getByRole("button", { name: "Create account" }));

    await waitFor(() => expect(onRegister).toHaveBeenCalledWith("new@example.com", "correctpassword1"));
  });

  it("shows the real backend error message on a failed login, not a generic one", async () => {
    const onLogin = vi.fn().mockRejectedValue(new Error("Incorrect email or password."));
    render(<AuthScreen onLogin={onLogin} onRegister={vi.fn()} />);

    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "a@example.com" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "wrongpassword" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));

    expect(await screen.findByText("Incorrect email or password.")).toBeInTheDocument();
  });

  it("disables the submit button while the request is in flight", async () => {
    let resolveLogin;
    const onLogin = vi.fn(() => new Promise((resolve) => { resolveLogin = resolve; }));
    render(<AuthScreen onLogin={onLogin} onRegister={vi.fn()} />);

    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "a@example.com" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "correctpassword1" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));

    expect(screen.getByRole("button", { name: "Please wait…" })).toBeDisabled();
    resolveLogin();
    await waitFor(() => expect(onLogin).toHaveBeenCalled());
  });
});
