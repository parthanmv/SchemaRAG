import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import type { GeneratedSQL, QueryResult } from "./api/types";

// ---------------------------------------------------------------------------
// fetch mocking - no live Gemini, PostgreSQL or backend involved.
// ---------------------------------------------------------------------------
type FetchHandler = (url: string, init?: RequestInit) => Promise<Response> | Response;

let handler: FetchHandler;
let fetchMock: ReturnType<typeof vi.fn>;

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
    headers: new Headers({ "content-type": "application/json" }),
  } as unknown as Response;
}

function stubFetch() {
  fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    return Promise.resolve(handler(url, init));
  });
  vi.stubGlobal("fetch", fetchMock);
}

beforeEach(() => {
  handler = (url) => {
    if (url.endsWith("/health")) {
      return jsonResponse(200, { status: "healthy", database: "connected", detail: null });
    }
    throw new Error(`Unexpected fetch: ${url}`);
  };
  stubFetch();
});

// ---------------------------------------------------------------------------
// Fixtures matching the real backend contracts.
// ---------------------------------------------------------------------------
const GENERATION_OK: GeneratedSQL = {
  question: "Which department has the highest average marks?",
  sql: "SELECT d.department_name, AVG(m.marks) AS average_marks\nFROM departments d\nJOIN students s ON s.department_id = d.department_id\nJOIN marks m ON m.student_id = s.student_id\nGROUP BY d.department_name\nORDER BY average_marks DESC\nLIMIT 1",
  model: "gemini-2.5-flash",
  grounded: true,
  retrieved_documents: [
    "schema_departments",
    "relationship_students_departments",
    "query_example_dept_highest_avg_marks",
  ],
  retrieval_scores: [0.91, 0.84, 0.77],
  issues: [],
  error: null,
};

const QUERY_OK: QueryResult = {
  question: "How many students are there?",
  sql: 'SELECT COUNT(*) AS student_count FROM "students"',
  model: "gemini-2.5-flash",
  grounded: true,
  security_allowed: true,
  security_issues: [],
  execution_status: "success",
  columns: ["student_count"],
  rows: [[1000]],
  row_count: 1,
  execution_time_ms: 12.4,
  retrieved_documents: ["schema_students"],
  error: null,
  executed_as: "schemarag_reader",
};

const HEALTH_UP = { status: "healthy", database: "connected", detail: null };

async function typeQuestion(text: string) {
  const user = userEvent.setup();
  await user.type(screen.getByLabelText(/ask a question/i), text);
  return user;
}

describe("SchemaRAG App", () => {
  it("renders header, question input, examples and health status", async () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: "SchemaRAG" })).toBeInTheDocument();
    expect(screen.getByLabelText(/ask a question/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /example questions|highest average marks/i }) ||
        screen.getAllByRole("button").length,
    ).toBeTruthy();
    expect(screen.getByText(/example questions/i)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText(/Backend connected/i)).toBeInTheDocument());
  });

  it("populates the input when an example question is clicked", async () => {
    render(<App />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Show the top 5 students by marks." }));
    expect(
      screen.getByLabelText(/ask a question/i) as HTMLTextAreaElement,
    ).toHaveValue("Show the top 5 students by marks.");
  });

  it("disables action buttons for too-short questions", async () => {
    render(<App />);
    // Settle the async /health update before asserting on static UI.
    await screen.findByText(/Backend connected/i);
    expect(screen.getByTestId("generate-button")).toBeDisabled();
    expect(screen.getByTestId("execute-button")).toBeDisabled();
  });

  it("clears state via the Clear button", async () => {
    render(<App />);
    const user = await typeQuestion("How many students are there?");
    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.getByLabelText(/ask a question/i)).toHaveValue("");
  });

  it("generates SQL without executing it", async () => {
    handler = (url) =>
      url.endsWith("/api/generate-sql")
        ? jsonResponse(200, GENERATION_OK)
        : url.endsWith("/health")
          ? jsonResponse(200, HEALTH_UP)
          : jsonResponse(500, {});
    render(<App />);
    const user = await typeQuestion("Which department has the highest average marks?");
    await user.click(screen.getByTestId("generate-button"));

    expect(await screen.findByTestId("sql-viewer")).toHaveTextContent(/SELECT/i);
    expect(screen.getByText("Grounded")).toBeInTheDocument();
    expect(screen.getByText("gemini-2.5-flash")).toBeInTheDocument();
    expect(screen.getByText(/Retrieved documents \(3\)/i)).toBeInTheDocument();

    const calls = fetchMock.mock.calls;
    expect(calls.some((c) => String(c[0]).endsWith("/api/query"))).toBe(false);
  });

  it("shows a loading state while generating SQL", async () => {
    let resolveGen!: (v: Response) => void;
    handler = (url) => {
      if (url.endsWith("/api/generate-sql")) {
        return new Promise<Response>((resolve) => {
          resolveGen = resolve;
        }) as unknown as Response;
      }
      return jsonResponse(200, HEALTH_UP);
    };
    render(<App />);
    const user = await typeQuestion("How many students are there?");
    await user.click(screen.getByTestId("generate-button"));
    expect(await screen.findByTestId("generating-indicator")).toBeInTheDocument();
    resolveGen(jsonResponse(200, GENERATION_OK));
    await waitFor(() =>
      expect(screen.queryByTestId("generating-indicator")).not.toBeInTheDocument(),
    );
  });

  it("executes a query through POST /api/query and renders results", async () => {
    handler = (url) =>
      url.endsWith("/api/query")
        ? jsonResponse(200, QUERY_OK)
        : url.endsWith("/health")
          ? jsonResponse(200, HEALTH_UP)
          : jsonResponse(500, {});
    render(<App />);
    const user = await typeQuestion("How many students are there?");
    await user.click(screen.getByTestId("execute-button"));

    expect(await screen.findByTestId("results-table")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "student_count" })).toBeInTheDocument();
    expect(screen.getByText("1000")).toBeInTheDocument();
    expect(screen.getByText("Approved")).toBeInTheDocument();
    expect(screen.getByText("Grounded")).toBeInTheDocument();
    expect(screen.getByText("schemarag_reader")).toBeInTheDocument();
    expect(screen.getByText(/12\.4 ms/)).toBeInTheDocument();
    const genCalls = fetchMock.mock.calls.filter((c) =>
      String(c[0]).endsWith("/api/generate-sql"),
    );
    expect(genCalls).toHaveLength(0);
  });

  it("renders an empty result set gracefully", async () => {
    handler = (url) =>
      url.endsWith("/api/query")
        ? jsonResponse(200, { ...QUERY_OK, execution_status: "empty_result", rows: [], row_count: 0 })
        : jsonResponse(200, HEALTH_UP);
    render(<App />);
    const user = await typeQuestion("How many students are there?");
    await user.click(screen.getByTestId("execute-button"));
    expect(await screen.findByText(/returned no rows/i)).toBeInTheDocument();
  });

  it("renders NULL values distinctly", async () => {
    handler = (url) =>
      url.endsWith("/api/query")
        ? jsonResponse(200, { ...QUERY_OK, columns: ["a", "b"], rows: [[1, null]], row_count: 1 })
        : jsonResponse(200, HEALTH_UP);
    render(<App />);
    const user = await typeQuestion("How many students are there?");
    await user.click(screen.getByTestId("execute-button"));
    expect(await screen.findByText("NULL")).toBeInTheDocument();
  });

  it("displays Not grounded when grounding fails", async () => {
    handler = (url) =>
      url.endsWith("/api/generate-sql")
        ? jsonResponse(200, { ...GENERATION_OK, grounded: false, sql: null })
        : jsonResponse(200, HEALTH_UP);
    render(<App />);
    const user = await typeQuestion("What is the meaning of life?");
    await user.click(screen.getByTestId("generate-button"));
    expect(await screen.findByText("Not grounded")).toBeInTheDocument();
    expect(screen.getByText(/No SQL was generated/i)).toBeInTheDocument();
  });

  it("maps HTTP 403 to the security rejection message", async () => {
    handler = (url) =>
      url.endsWith("/api/query")
        ? jsonResponse(403, {
            detail: { error: "security_rejected", issues: ["only SELECT/WITH read queries are allowed"] },
          })
        : jsonResponse(200, HEALTH_UP);
    render(<App />);
    const user = await typeQuestion("Delete all students");
    await user.click(screen.getByTestId("execute-button"));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/rejected by the SQL security layer/i);
    // Backend-provided reasons surface in the dedicated rejection panel.
    expect(screen.getByTestId("security-issues")).toHaveTextContent(
      /only SELECT\/WITH read queries are allowed/i,
    );
  });

  it("maps HTTP 400 insufficient_context to its friendly message", async () => {
    handler = (url) =>
      url.endsWith("/api/generate-sql")
        ? jsonResponse(400, { detail: { error: "insufficient_context", issues: [] } })
        : jsonResponse(200, HEALTH_UP);
    render(<App />);
    const user = await typeQuestion("Tell me something interesting");
    await user.click(screen.getByTestId("generate-button"));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /Schema information is insufficient/i,
    );
  });

  it("maps HTTP 502 to the LLM-unavailable message", async () => {
    handler = (url) =>
      url.endsWith("/api/generate-sql")
        ? jsonResponse(502, {})
        : jsonResponse(200, HEALTH_UP);
    render(<App />);
    const user = await typeQuestion("How many students are there?");
    await user.click(screen.getByTestId("generate-button"));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /SQL generation service is currently unavailable/i,
    );
  });

  it("reports backend unavailability on network failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))),
    );
    render(<App />);
    const user = await typeQuestion("How many students are there?");
    await user.click(screen.getByTestId("execute-button"));
    expect(await screen.findByRole("alert")).toHaveTextContent(/Backend is unavailable/i);
    expect(screen.getByText("Backend is unavailable.", { selector: "[role=status]" })).toBeInTheDocument();
  });

  it("shows the database-unavailable state when /health returns 503", async () => {
    handler = (url) =>
      url.endsWith("/health")
        ? jsonResponse(503, { status: "unhealthy", database: "unavailable", detail: "down" })
        : jsonResponse(500, {});
    render(<App />);
    // Backend answered, so only the database side is reported as down.
    expect(
      await screen.findByText(/Backend connected · Database unavailable/),
    ).toBeInTheDocument();
  });

  it("shows a Rejected security verdict with backend reasons on HTTP 403", async () => {
    handler = (url) =>
      url.endsWith("/api/query")
        ? jsonResponse(403, {
            detail: { error: "security_rejected", issues: ["unknown table: users"] },
          })
        : jsonResponse(200, HEALTH_UP);
    render(<App />);
    const user = await typeQuestion("Show me the users table");
    await user.click(screen.getByTestId("execute-button"));
    expect(await screen.findByText("Rejected")).toBeInTheDocument();
    expect(screen.getByTestId("security-issues")).toHaveTextContent(
      "unknown table: users",
    );
  });

  it("maps generation error codes in 200 payloads to friendly messages", async () => {
    handler = (url) =>
      url.endsWith("/api/generate-sql")
        ? jsonResponse(200, {
            ...GENERATION_OK,
            sql: null,
            grounded: false,
            error: "insufficient_context",
          })
        : jsonResponse(200, HEALTH_UP);
    render(<App />);
    const user = await typeQuestion("Tell me something interesting");
    await user.click(screen.getByTestId("generate-button"));
    expect(
      await screen.findByText(
        /Schema information is insufficient to generate a reliable query/i,
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText("insufficient_context")).not.toBeInTheDocument();
  });

  it("survives malformed JSON responses", async () => {
    handler = (url) =>
      url.endsWith("/api/query")
        ? ({
            ok: true,
            status: 200,
            json: () => Promise.reject(new SyntaxError("bad json")),
            headers: new Headers(),
          } as unknown as Response)
        : jsonResponse(200, HEALTH_UP);
    render(<App />);
    const user = await typeQuestion("How many students are there?");
    await user.click(screen.getByTestId("execute-button"));
    expect(await screen.findByRole("alert")).toHaveTextContent(/unexpected response/i);
  });
});
