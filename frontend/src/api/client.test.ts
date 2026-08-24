import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, executeQuery, generateSql, getHealth } from "./client";

// ---------------------------------------------------------------------------
// fetch mocking - no live backend involved.
// ---------------------------------------------------------------------------
function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
    headers: new Headers({ "content-type": "application/json" }),
  } as unknown as Response;
}

function stubFetch(impl: () => Promise<Response> | Response) {
  vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(impl())));
}

beforeEach(() => {
  stubFetch(() =>
    jsonResponse(200, { status: "healthy", database: "connected", detail: null }),
  );
});

describe("API client", () => {
  it("parses a healthy /health body", async () => {
    const health = await getHealth();
    expect(health).toEqual({
      status: "healthy",
      database: "connected",
      detail: null,
    });
  });

  it("returns the /health body even when the endpoint answers 503", async () => {
    stubFetch(() =>
      jsonResponse(503, {
        status: "unhealthy",
        database: "unavailable",
        detail: "PostgreSQL is unreachable.",
      }),
    );
    const health = await getHealth();
    expect(health.database).toBe("unavailable");
    expect(health.status).toBe("unhealthy");
  });

  it("rejects with kind=network when fetch fails outright", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))),
    );
    await expect(getHealth()).rejects.toMatchObject({
      kind: "network",
      message: "Backend is unavailable.",
    });
  });

  it("rejects with kind=timeout when the request aborts", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        () =>
          new Promise((_resolve, reject) =>
            setTimeout(
              () => reject(new DOMException("Aborted", "AbortError")),
              10,
            ),
          ),
      ),
    );
    await expect(getHealth(5)).rejects.toMatchObject({ kind: "timeout" });
  });

  it("sends POST JSON to /api/generate-sql and parses the payload", async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve(
        jsonResponse(200, {
          question: "q?",
          sql: "SELECT 1",
          model: "gemini:test",
          grounded: true,
          retrieved_documents: ["schema_students"],
          retrieval_scores: [0.9],
          issues: [],
          error: null,
        }),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);
    const gen = await generateSql("How many students are there?");
    expect(gen.sql).toBe("SELECT 1");
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toContain("/api/generate-sql");
    expect(init.method).toBe("POST");
    expect(init.body).toBe(JSON.stringify({ question: "How many students are there?" }));
  });

  it("maps generate-sql 503 to the LLM-unavailable message", async () => {
    stubFetch(() => jsonResponse(503, { detail: "LLM backend is unavailable; ..." }));
    await expect(generateSql("question")).rejects.toMatchObject({
      kind: "unavailable",
      message: "SQL generation service is currently unavailable.",
    });
  });

  it("maps generate-sql 502 to the LLM-unavailable message", async () => {
    stubFetch(() => jsonResponse(502, { detail: "boom" }));
    await expect(generateSql("question")).rejects.toMatchObject({
      kind: "llm_error",
      message: "SQL generation service is currently unavailable.",
    });
  });

  it("maps 400 insufficient_context to the insufficient-context message", async () => {
    stubFetch(() =>
      jsonResponse(400, { detail: { error: "insufficient_context", issues: [] } }),
    );
    await expect(executeQuery("question")).rejects.toMatchObject({
      message: "Schema information is insufficient to generate a reliable query.",
    });
  });

  it("maps other 400s to the grounding-failure message and carries issues", async () => {
    stubFetch(() =>
      jsonResponse(400, {
        detail: { error: "not_grounded", issues: ["unknown table: users"] },
      }),
    );
    const err = await executeQuery("question").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.kind).toBe("bad_request");
    expect(err.message).toBe("Generated SQL references unknown database objects.");
    expect(err.issues).toEqual(["unknown table: users"]);
  });

  it("maps 403 to security_rejected with backend reasons attached", async () => {
    stubFetch(() =>
      jsonResponse(403, {
        detail: {
          error: "security_rejected",
          issues: ["only SELECT/WITH read queries are allowed"],
        },
      }),
    );
    const err = await executeQuery("DROP TABLE students").catch((e) => e);
    expect(err.kind).toBe("security_rejected");
    expect(err.message).toBe("This query was rejected by the SQL security layer.");
    expect(err.issues).toEqual(["only SELECT/WITH read queries are allowed"]);
  });

  it("maps query 503 (database down, not LLM) to execution failure", async () => {
    stubFetch(() =>
      jsonResponse(503, { detail: "could not reach the database with the execution role" }),
    );
    await expect(executeQuery("question")).rejects.toMatchObject({
      kind: "server",
      message: "Query execution failed.",
    });
  });

  it("maps query 503 with an LLM detail to the LLM-unavailable message", async () => {
    stubFetch(() =>
      jsonResponse(503, { detail: "LLM backend is unavailable; check settings." }),
    );
    await expect(executeQuery("question")).rejects.toMatchObject({
      message: "SQL generation service is currently unavailable.",
    });
  });

  it("maps query 504 (statement timeout) to execution failure", async () => {
    stubFetch(() => jsonResponse(504, { detail: "statement exceeded timeout" }));
    await expect(executeQuery("question")).rejects.toMatchObject({
      message: "Query execution failed.",
    });
  });

  it("maps query 500 to execution failure", async () => {
    stubFetch(() => jsonResponse(500, { detail: null }));
    await expect(executeQuery("question")).rejects.toMatchObject({
      message: "Query execution failed.",
    });
  });

  it("summarises FastAPI 422 validation errors", async () => {
    stubFetch(() =>
      jsonResponse(422, {
        detail: [
          { msg: "String should have at least 3 characters" },
          { msg: "String should have at most 500 characters" },
        ],
      }),
    );
    const err = await executeQuery("a").catch((e) => e);
    expect(err.kind).toBe("validation");
    expect(err.message).toContain("at least 3 characters");
  });

  it("flags malformed success payloads", async () => {
    stubFetch(() => ({
      ok: true,
      status: 200,
      json: () => Promise.reject(new SyntaxError("bad json")),
    }) as unknown as Response);
    await expect(executeQuery("question")).rejects.toMatchObject({
      kind: "malformed",
    });
  });
});
