import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ComposerMenu } from "./ComposerMenu";

describe("ComposerMenu", () => {
  it("starts closed — no menu items visible until the + button is clicked", () => {
    render(<ComposerMenu onSelect={vi.fn()} />);
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });

  it("opens the menu and shows every action group on clicking +", async () => {
    render(<ComposerMenu onSelect={vi.fn()} />);
    await userEvent.click(screen.getByLabelText("More actions"));
    expect(screen.getByRole("menu")).toBeInTheDocument();
    expect(screen.getByText("Research")).toBeInTheDocument();
    expect(screen.getByText("Learning & Interview")).toBeInTheDocument();
    expect(screen.getByText("Analysis")).toBeInTheDocument();
    expect(screen.getByText("Evaluation")).toBeInTheDocument();
    expect(screen.getByText("Deep Research")).toBeInTheDocument();
    expect(screen.getByText("Project Interview")).toBeInTheDocument();
    expect(screen.getByText("Knowledge Graph")).toBeInTheDocument();
    expect(screen.getByText("Eval Dashboard")).toBeInTheDocument();
  });

  it("calls onSelect with the action's key and closes the menu", async () => {
    const onSelect = vi.fn();
    render(<ComposerMenu onSelect={onSelect} />);
    await userEvent.click(screen.getByLabelText("More actions"));
    await userEvent.click(screen.getByText("Compare Papers"));
    expect(onSelect).toHaveBeenCalledWith("compare");
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });

  it("closes the menu when the backdrop is clicked, without calling onSelect", async () => {
    const onSelect = vi.fn();
    const { container } = render(<ComposerMenu onSelect={onSelect} />);
    await userEvent.click(screen.getByLabelText("More actions"));
    const backdrop = container.querySelector(".composer-menu-backdrop");
    await userEvent.click(backdrop);
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    expect(onSelect).not.toHaveBeenCalled();
  });
});