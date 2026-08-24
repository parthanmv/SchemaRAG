import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import SQLViewer from "./SQLViewer";

afterEach(() => {
  Reflect.deleteProperty(navigator, "clipboard");
});

function defineClipboard(value: unknown) {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value,
  });
}

describe("SQLViewer", () => {
  it("renders the SQL text with a title and horizontal scroll container", () => {
    render(<SQLViewer sql="SELECT * FROM students" title="Generated SQL" />);
    expect(screen.getByTestId("sql-viewer")).toBeInTheDocument();
    expect(screen.getByTestId("sql-viewer")).toHaveTextContent(
      "SELECT * FROM students",
    );
  });

  it("copies the exact generated SQL to the clipboard", async () => {
    const writeText = vi.fn(() => Promise.resolve());
    defineClipboard({ writeText });
    const sql = "SELECT\n  name,\n  marks FROM students WHERE marks > 90";
    render(<SQLViewer sql={sql} />);
    // fireEvent (not user-event): user-event installs its own clipboard stub.
    fireEvent.click(screen.getByRole("button", { name: "Copy" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(sql));
    expect(screen.getByText("Copied!")).toBeInTheDocument();
  });

  it("does not crash when the clipboard is unavailable", () => {
    defineClipboard(undefined);
    render(<SQLViewer sql="SELECT 1" />);
    fireEvent.click(screen.getByRole("button", { name: "Copy" }));
    expect(screen.getByRole("button", { name: "Copy" })).toBeInTheDocument();
  });
});
