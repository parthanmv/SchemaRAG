import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import ResultsTable from "./ResultsTable";
import { GenerationMeta } from "./RetrievalPanel";
import type { GeneratedSQL } from "../api/types";

describe("ResultsTable column_kinds (Phase 7)", () => {
  const columns = ["department_name", "budget", "is_active"];

  it("right-aligns numeric and boolean columns when kinds are provided", () => {
    render(
      <table>
        <ResultsTable
          columns={columns}
          rows={[["CSE", 120000, true]]}
          columnKinds={["text", "number", "boolean"]}
        />
      </table>,
    );
    const budgetHeader = screen.getByText("budget").closest("th");
    expect(budgetHeader?.className).toContain("text-right");
    const budgetCell = screen.getByText("120000").closest("td");
    expect(budgetCell?.className).toContain("text-right");
    expect(budgetCell?.className).toContain("tabular-nums");
    const activeCell = screen.getByText("true").closest("td");
    expect(activeCell?.className).toContain("text-right");
    // Text columns stay left-aligned.
    const nameCell = screen.getByText("CSE").closest("td");
    expect(nameCell?.className).not.toContain("text-right");
  });

  it("keeps default left alignment without kinds (backwards compatible)", () => {
    render(<ResultsTable columns={columns} rows={[["CSE", 1, false]]} />);
    const budgetCell = screen.getByText("1").closest("td");
    expect(budgetCell?.className).not.toContain("tabular-nums");
  });
});

describe("GenerationMeta processed question (Phase 7)", () => {
  const base: GeneratedSQL = {
    question: "Which departments have budgets?",
    processed_question: "which departments have budgets?",
    sql: "SELECT department_name FROM departments",
    model: "gemini:test",
    grounded: true,
    retrieved_documents: ["schema_departments"],
    retrieval_scores: [0.9],
    issues: [],
    error: null,
  };

  it("shows the preprocessed form when it differs from the original", () => {
    render(<GenerationMeta gen={base} />);
    expect(screen.getByTestId("processed-question")).toHaveTextContent(
      "Preprocessed:",
    );
    expect(screen.getByTestId("processed-question")).toHaveTextContent(
      "which departments have budgets?",
    );
  });

  it("hides the line when no preprocessing changed anything", () => {
    render(<GenerationMeta gen={{ ...base, processed_question: null }} />);
    expect(screen.queryByTestId("processed-question")).not.toBeInTheDocument();
  });
});
