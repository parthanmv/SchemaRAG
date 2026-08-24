import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import RetrievalPanel from "./RetrievalPanel";

describe("RetrievalPanel", () => {
  it("renders nothing when there are no documents", () => {
    render(<RetrievalPanel documents={[]} />);
    expect(screen.queryByTestId("retrieval-panel")).not.toBeInTheDocument();
  });

  it("groups documents by type prefix including multi-word prefixes", async () => {
    const user = userEvent.setup();
    render(
      <RetrievalPanel
        documents={[
          "schema_students",
          "relationship_marks_courses",
          "constraint_check_marks_range",
          "business_rule_attendance",
          "query_example_top_students",
        ]}
        scores={[0.91, 0.82, 0.7, 0.64, 0.55]}
      />,
    );
    await user.click(screen.getByText(/Retrieved documents \(5\)/i));
    expect(screen.getByText("Schema")).toBeInTheDocument();
    expect(screen.getByText("Relationships")).toBeInTheDocument();
    expect(screen.getByText("Constraints")).toBeInTheDocument();
    expect(screen.getByText("Business rules")).toBeInTheDocument();
    expect(screen.getByText("Query examples")).toBeInTheDocument();
    // No stray "Other" bucket when every id matches a known prefix.
    expect(screen.queryByText("Other")).not.toBeInTheDocument();
    expect(screen.getByText("schema_students")).toBeInTheDocument();
    expect(screen.getByText(/score 0\.9100/)).toBeInTheDocument();
  });

  it("falls back to Other for unrecognised ids", async () => {
    const user = userEvent.setup();
    render(<RetrievalPanel documents={["mystery_doc"]} />);
    await user.click(screen.getByText(/Retrieved documents \(1\)/i));
    expect(screen.getByText("Other")).toBeInTheDocument();
  });
});
